import asyncio
import json
import logging
import os
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from contextlib import asynccontextmanager
import config

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def translate_openai(text, target_lang, model):
    from openai import OpenAI
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"Sen profesyonel bir simultane tercümansın. Verilen metni {target_lang} diline doğal ve akıcı çevir. Sadece çeviriyi yaz."},
            {"role": "user", "content": text}
        ],
        max_tokens=400
    )
    return r.choices[0].message.content.strip()

def translate_anthropic(text, target_lang, model):
    import anthropic
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=model, max_tokens=400,
        messages=[{"role": "user", "content": f"Translate to {target_lang}, only the translation:\n\n{text}"}]
    )
    return msg.content[0].text.strip()

def translate_ollama(text, target_lang):
    import httpx
    r = httpx.post(f"{config.OLLAMA_URL}/api/generate", json={
        "model": config.OLLAMA_MODEL,
        "prompt": f"Translate to {target_lang}, reply with ONLY the translation:\n\n{text}",
        "stream": False
    }, timeout=30)
    return r.json()["response"].strip()

def translate(text, target_lang, tr_engine, tr_model):
    try:
        if tr_engine == "openai":
            return translate_openai(text, target_lang, tr_model)
        elif tr_engine == "anthropic":
            return translate_anthropic(text, target_lang, tr_model)
        elif tr_engine == "ollama":
            return translate_ollama(text, target_lang)
    except Exception as e:
        logger.error(f"Çeviri hatası ({tr_engine}): {e}")
        return ""

async def run_deepgram(client_ws, stt_lang, tr_engine, tr_model, target_lang):
    SENTENCE_ENDINGS = {'.', '?', '!'}
    MAX_WAIT = 20
    utterance_buffer = []
    flush_timer_task = None
    translation_queue = asyncio.Queue()

    url = (
        "wss://api.deepgram.com/v1/listen"
        f"?model=nova-3&language={stt_lang}"
        "&interim_results=true&smart_format=true&punctuate=true"
        "&endpointing=300&utterance_end_ms=1200"
    )
    headers = {"Authorization": f"Token {config.DEEPGRAM_API_KEY}"}

    async def translation_worker():
        while True:
            item = await translation_queue.get()
            if item is None:
                break
            text, tgt = item
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, translate, text, tgt, tr_engine, tr_model
                )
                if result:
                    await client_ws.send_json({
                        "type": "translation",
                        "english": text,
                        "turkish": result
                    })
            except Exception as e:
                logger.error(f"Worker hatası: {e}")
            finally:
                translation_queue.task_done()

    async def flush_buffer():
        nonlocal utterance_buffer, flush_timer_task
        if utterance_buffer:
            full_text = " ".join(utterance_buffer).strip()
            utterance_buffer = []
            if full_text and len(full_text.split()) >= 2:
                await translation_queue.put((full_text, target_lang))
        if flush_timer_task and not flush_timer_task.done():
            flush_timer_task.cancel()
        flush_timer_task = None

    def reset_timer():
        nonlocal flush_timer_task
        if flush_timer_task and not flush_timer_task.done():
            flush_timer_task.cancel()
        async def timer():
            await asyncio.sleep(MAX_WAIT)
            if utterance_buffer:
                await flush_buffer()
        flush_timer_task = asyncio.create_task(timer())

    tr_task = asyncio.create_task(translation_worker())

    try:
        async with websockets.connect(url, additional_headers=headers, ping_interval=20, ping_timeout=60) as dg_ws:

            async def receive_from_client():
                try:
                    while True:
                        data = await client_ws.receive_bytes()
                        if dg_ws.state.value == 1:
                            await dg_ws.send(data)
                except WebSocketDisconnect:
                    pass
                except Exception as e:
                    logger.error(f"Client ses hatası: {e}")
                finally:
                    try:
                        await dg_ws.send(json.dumps({"type": "CloseStream"}))
                        await asyncio.sleep(0.3)
                    except:
                        pass

            async def receive_from_deepgram():
                nonlocal utterance_buffer, flush_timer_task
                try:
                    async for message in dg_ws:
                        try:
                            data = json.loads(message)
                        except:
                            continue
                        msg_type = data.get("type", "")
                        if msg_type == "Results":
                            alts = data.get("channel", {}).get("alternatives", [{}])
                            if not alts:
                                continue
                            transcript = alts[0].get("transcript", "").strip()
                            is_final = data.get("is_final", False)
                            speech_final = data.get("speech_final", False)
                            if not transcript:
                                continue
                            if not is_final:
                                display = (" ".join(utterance_buffer) + " " + transcript).strip()
                                await client_ws.send_json({"type": "partial", "text": display})
                            else:
                                utterance_buffer.append(transcript)
                                display = " ".join(utterance_buffer).strip()
                                await client_ws.send_json({"type": "partial", "text": display})
                                ends = bool(transcript.strip()) and transcript.strip()[-1] in SENTENCE_ENDINGS
                                if ends or speech_final:
                                    await flush_buffer()
                                else:
                                    reset_timer()
                        elif msg_type == "UtteranceEnd":
                            if utterance_buffer:
                                await flush_buffer()
                except Exception as e:
                    logger.error(f"Deepgram hatası: {e}")
                finally:
                    if flush_timer_task and not flush_timer_task.done():
                        flush_timer_task.cancel()
                    if utterance_buffer:
                        await flush_buffer()
                    await translation_queue.put(None)

            await asyncio.gather(receive_from_client(), receive_from_deepgram())

    except Exception as e:
        logger.error(f"Deepgram bağlantı hatası: {e}")
        await translation_queue.put(None)

    await tr_task

async def run_whisper(client_ws, stt_lang, tr_engine, tr_model, target_lang):
    import numpy as np
    import subprocess
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from faster_whisper import WhisperModel
    model = WhisperModel(config.WHISPER_MODEL, device=config.WHISPER_DEVICE, compute_type=config.WHISPER_COMPUTE_TYPE)
    translation_queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    audio_buffer = bytearray()
    CHUNK_SIZE = 100000
    last_text = ""

    def webm_to_pcm(webm_bytes):
        try:
            proc = subprocess.run(
                ["ffmpeg", "-i", "pipe:0", "-f", "f32le", "-ar", "16000", "-ac", "1", "pipe:1"],
                input=webm_bytes, capture_output=True, timeout=10
            )
            if proc.returncode != 0:
                return None
            return np.frombuffer(proc.stdout, dtype=np.float32)
        except Exception as e:
            logger.error(f"FFmpeg hatası: {e}")
            return None

    async def translation_worker():
        while True:
            item = await translation_queue.get()
            if item is None:
                break
            try:
                result = await loop.run_in_executor(None, translate, item, target_lang, tr_engine, tr_model)
                if result:
                    await client_ws.send_json({"type": "translation", "english": item, "turkish": result})
            except Exception as e:
                logger.error(f"Worker hatası: {e}")
            finally:
                translation_queue.task_done()

    def transcribe_chunk(audio_bytes):
        nonlocal last_text
        audio_np = webm_to_pcm(audio_bytes)
        if audio_np is None or len(audio_np) == 0:
            return ""
        if np.abs(audio_np).mean() < 0.01:
            return ""
        lang_code = stt_lang.split("-")[0]
        segments, _ = model.transcribe(audio_np, language=lang_code, vad_filter=True)
        text = " ".join([s.text for s in segments]).strip()
        if text == last_text:
            return ""
        last_text = text
        return text

    tr_task = asyncio.create_task(translation_worker())
    try:
        while True:
            try:
                data = await client_ws.receive_bytes()
                audio_buffer.extend(data)
                if len(audio_buffer) >= CHUNK_SIZE:
                    chunk = bytes(audio_buffer)
                    audio_buffer.clear()
                    text = await loop.run_in_executor(None, transcribe_chunk, chunk)
                    if text:
                        await client_ws.send_json({"type": "partial", "text": text})
                        await translation_queue.put(text)
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Whisper receive hatası: {e}")
                break
    finally:
        await translation_queue.put(None)
        await tr_task

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=config.SECRET_KEY)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def is_authenticated(request: Request) -> bool:
    return request.session.get("authenticated") is True

@app.get("/login")
async def login_page(request: Request, error: str = ""):
    with open("templates/login.html", "r", encoding="utf-8") as f:
        html = f.read()
    if error:
        html = html.replace("<!--ERROR-->", '<div class="login-error">Kullanıcı adı veya şifre hatalı</div>')
    return HTMLResponse(html)

@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if username == config.AUTH_USERNAME and password == config.AUTH_PASSWORD:
        request.session["authenticated"] = True
        return RedirectResponse(url="/", status_code=303)
    return RedirectResponse(url="/login?error=1", status_code=303)

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)

@app.get("/")
async def get(request: Request):
    if not is_authenticated(request):
        return RedirectResponse(url="/login", status_code=303)
    with open("templates/panel.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/config")
async def get_config(request: Request):
    if not is_authenticated(request):
        return RedirectResponse(url="/login", status_code=303)
    return {
        "allow_local": config.ALLOW_LOCAL,
        "stt_languages": config.STT_LANGUAGES,
        "translation_languages": config.TRANSLATION_LANGUAGES,
        "openai_models": config.OPENAI_MODELS,
        "anthropic_models": config.ANTHROPIC_MODELS,
        "ollama_model": config.OLLAMA_MODEL,
    }

@app.websocket("/asr")
async def websocket_endpoint(client_ws: WebSocket):
    await client_ws.accept()
    try:
        init_msg = await asyncio.wait_for(client_ws.receive_text(), timeout=10)
        params = json.loads(init_msg)
    except Exception:
        await client_ws.send_json({"type": "error", "msg": "Config alınamadı"})
        return

    stt_engine = params.get("stt_engine", "deepgram")
    stt_lang = params.get("stt_lang", "en-US")
    tr_engine = params.get("tr_engine", "openai")
    tr_model = params.get("tr_model", "gpt-4o-mini")
    target_lang = params.get("target_lang", "Türkçe")

    await client_ws.send_json({"type": "status", "msg": f"Başlatılıyor... {stt_engine}"})

    try:
        if stt_engine == "deepgram":
            await run_deepgram(client_ws, stt_lang, tr_engine, tr_model, target_lang)
        elif stt_engine == "whisper" and config.ALLOW_LOCAL:
            await run_whisper(client_ws, stt_lang, tr_engine, tr_model, target_lang)
        else:
            await client_ws.send_json({"type": "error", "msg": "Geçersiz motor seçimi"})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"ASR hatası: {e}")
        try:
            await client_ws.send_json({"type": "error", "msg": str(e)})
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    print("SimultanePro başlatılıyor...")
    print(f"Local mod: {config.ALLOW_LOCAL}")
    print(f"http://localhost:{config.PORT}")
    uvicorn.run(app, host=config.HOST, port=config.PORT)
