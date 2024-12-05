import asyncio
import websockets
import json
import base64
import numpy as np
from faster_whisper import WhisperModel
from TTS.api import TTS
import webrtcvad
import wave
import os
import tempfile
from datetime import datetime
from openai import AsyncOpenAI
from collections import deque
import time
from rich.console import Console
import re
import struct
import torch
print(torch.cuda.is_available())


class VoiceAssistantServer:
    def __init__(self):
        self.console = Console()
        self.console.print("Initializing Voice Assistant Server...", style="bold blue")
        
        # Audio settings
        self.RATE = 16000
        self.CHANNELS = 1
        self.CHUNK_MS = 30
        self.CHUNK_SIZE = int(self.RATE * self.CHUNK_MS / 1000)
        
        # Initialize components
        self.active_connections = set()
        self.client_buffers = {}
        self.client_states = {}
        self.debug = True
        self.conversations = {} 
        
        # Interruption handling
        self.current_response_task = {}
        self.current_tts_task = {}
        self.is_speaking = {}
        self.interrupt_event = {}
        self.pending_speech = {}  # Track pending speech processing
        self.speech_confirmed = {}  # Track if speech was successfully transcribed
        # Processing locks
        self.speech_processing_lock = asyncio.Lock()

        # Load models
        self._load_models()
        
        # LLM Client setup
        self.llm_client = AsyncOpenAI(
            api_key='ollama',
            base_url='http://localhost:11434/v1'
        )
        
        # Debug flag
        

    def _load_models(self):
        try:
            self.console.print("Loading Whisper model...", style="yellow")
            self.whisper = WhisperModel("small", compute_type="int8")
            
            self.console.print("Loading TTS model...", style="yellow")
            self.tts_model = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=False, gpu=True)
            
            self.console.print("Initializing VAD...", style="yellow")
            self.vad = webrtcvad.Vad(2)
            
            self.console.print("All models loaded successfully!", style="bold green")
        except Exception as e:
            self.console.print(f"Error loading models: {e}", style="bold red")
            raise
        self.tts_temp_dir = tempfile.mkdtemp(prefix='tts_audio_')
        self.console.print(f"Created temp directory for TTS: {self.tts_temp_dir}", style="blue")
        
    def __del__(self):
        # Cleanup temp directory
        try:
            import shutil
            shutil.rmtree(self.tts_temp_dir)
        except Exception as e:
            if self.debug:
                self.console.print(f"Error cleaning up temp directory: {e}", style="red")
    def _init_client_state(self, websocket):
        self.client_buffers[websocket] = {
            'audio_buffer': deque(maxlen=50),
            'speech_buffer': [],
            'response_buffer': []
        }
        self.client_states[websocket] = {
            'silence_frames': 0,
            'is_speaking': False,
            'speech_detected': False,
            'last_process_time': time.time(),
            'conversation_history': []
        }
        self.current_response_task[websocket] = None
        self.current_tts_task[websocket] = None
        self.is_speaking[websocket] = False
        self.interrupt_event[websocket] = asyncio.Event()
        self.pending_speech[websocket] = False
        self.speech_confirmed[websocket] = False
        self.conversations[websocket] = [
            {
                "role": "system", 
                "content": (
                    "You are a professional HR representative from Thaagam Foundation conducting a voice interview "
                    "for a Voice Process position. Keep your responses concise, professional, and focused on "
                    "evaluating the candidate's skills and experience. Ask one question at a time and wait for "
                    "the candidate's response. Do not make up or assume the candidate's responses. Respond naturally "
                    "to what they actually say."
                )
            },
            {
                "role": "assistant",
                "content": (
                    "Hello! I'm the HR representative from Thaagam Foundation. Thank you for your interest "
                    "in our Voice Process position. Could you tell me about your relevant experience?"
                )
            }
        ]

    def _cleanup_client(self, websocket):
        if websocket in self.client_buffers:
            del self.client_buffers[websocket]
        if websocket in self.client_states:
            del self.client_states[websocket]
        if websocket in self.current_response_task:
            del self.current_response_task[websocket]
        if websocket in self.current_tts_task:
            del self.current_tts_task[websocket]
        if websocket in self.is_speaking:
            del self.is_speaking[websocket]
        if websocket in self.interrupt_event:
            del self.interrupt_event[websocket]
        self.active_connections.remove(websocket)
        if websocket in self.pending_speech:
            del self.pending_speech[websocket]
        if websocket in self.speech_confirmed:
            del self.speech_confirmed[websocket]
        if websocket in self.conversations:
            del self.conversations[websocket]


    async def _cancel_current_tasks(self, websocket):
        """Cancel tasks only if speech is confirmed"""
        if not self.speech_confirmed[websocket]:
            return

        try:
            if websocket in self.current_response_task and self.current_response_task[websocket]:
                self.current_response_task[websocket].cancel()
                self.current_response_task[websocket] = None

            if websocket in self.current_tts_task and self.current_tts_task[websocket]:
                self.current_tts_task[websocket].cancel()
                self.current_tts_task[websocket] = None

            self.interrupt_event[websocket].set()
            await asyncio.sleep(0.1)
            self.interrupt_event[websocket].clear()

            await websocket.send(json.dumps({
                'type': 'interrupt'
            }))

            self.is_speaking[websocket] = False

            if self.debug:
                self.console.print("Tasks cancelled due to confirmed speech", style="yellow")

        except Exception as e:
            if self.debug:
                self.console.print(f"Error in task cancellation: {e}", style="bold red")
        finally:
            # Reset speech states
            self.pending_speech[websocket] = False
            self.speech_confirmed[websocket] = False

    def _prepare_frame(self, audio_data):
        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            frame_size = int(self.RATE * 0.03)
            
            if len(audio_np) < frame_size:
                audio_np = np.pad(audio_np, (0, frame_size - len(audio_np)))
            elif len(audio_np) > frame_size:
                audio_np = audio_np[:frame_size]
            
            return audio_np.tobytes()
        except Exception as e:
            if self.debug:
                self.console.print(f"Error preparing frame: {e}", style="bold red")
            return None

    async def _process_audio_chunk(self, websocket, audio_data):
        """Process audio with smart interruption"""
        try:
            state = self.client_states[websocket]
            buffers = self.client_buffers[websocket]
            
            frame = self._prepare_frame(audio_data)
            if not frame:
                return
            
            is_speech = self.vad.is_speech(frame, self.RATE)
            
            if is_speech:
                state['silence_frames'] = 0
                if not state['speech_detected']:
                    state['speech_detected'] = True
                    self.pending_speech[websocket] = True  # Mark speech as pending
                    await websocket.send(json.dumps({
                        'type': 'status',
                        'message': 'Speech detected'
                    }))
                buffers['speech_buffer'].append(audio_data)
            else:
                if state['speech_detected']:
                    state['silence_frames'] += 1
                    buffers['speech_buffer'].append(audio_data)

                    if state['silence_frames'] > 15:  # ~450ms of silence
                        if len(buffers['speech_buffer']) > 0:
                            speech_data = b''.join(buffers['speech_buffer'])
                            buffers['speech_buffer'] = []
                            state['speech_detected'] = False
                            state['silence_frames'] = 0
                            # Process the speech
                            asyncio.create_task(self._process_speech_buffer(websocket, speech_data))
                        else:
                            # No valid speech detected
                            self.pending_speech[websocket] = False
                            self.speech_confirmed[websocket] = False

        except Exception as e:
            if self.debug:
                self.console.print(f"Error processing audio chunk: {e}", style="bold red")

    async def _process_speech_buffer(self, websocket, audio_data):
        """Process speech with transcription and response generation"""
        temp_file = None
        try:
            temp_file = tempfile.NamedTemporaryFile(
                suffix='.wav',
                prefix=f'speech_{time.time_ns()}_',
                delete=False
            ).name

            with wave.open(temp_file, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(self.RATE)
                wf.writeframes(audio_data)

            segments, _ = self.whisper.transcribe(temp_file, language="en")
            
            transcription_successful = False
            for segment in segments:
                text = segment.text.strip()
                if text and len(text) > 1:
                    transcription_successful = True
                    if self.debug:
                        self.console.print(f"Transcribed: {text}", style="green")
                    
                    # Store user's response in conversation history
                    self.conversations[websocket].append({
                        "role": "user",
                        "content": text
                    })
                    
                    # Send transcription to client
                    await websocket.send(json.dumps({
                        'type': 'transcription',
                        'text': text
                    }))
                    
                    # Generate and send response - removed text argument
                    self.current_response_task[websocket] = asyncio.create_task(
                        self._generate_response(websocket)
                    )

            if not transcription_successful:
                if self.debug:
                    self.console.print("No valid transcription", style="yellow")

        except Exception as e:
            if self.debug:
                self.console.print(f"Error processing speech buffer: {e}", style="bold red")
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    if self.debug:
                        self.console.print(f"Error cleaning up temp file: {e}", style="dim red")
    async def _generate_response(self, websocket):
        """Generate response using conversation history"""
        try:
            # Get prompt from conversation history
            prompt = self._create_prompt(websocket)
            
            stream = await self.llm_client.completions.create(
                model="mistral",
                prompt=prompt,
                temperature=0.7,
                max_tokens=30,
                stream=True
            )

            current_sentence = ""
            sentence_end_pattern = re.compile(r'[.!?]+\s*')
            full_response = ""

            async for chunk in stream:
                if self.interrupt_event[websocket].is_set():
                    return

                if not chunk.choices[0].text:
                    continue
                    
                current_sentence += chunk.choices[0].text
                
                match = sentence_end_pattern.search(current_sentence)
                if match:
                    end_idx = match.end()
                    complete_sentence = current_sentence[:end_idx].strip()
                    current_sentence = current_sentence[end_idx:].strip()
                    full_response += complete_sentence + " "

                    if complete_sentence and not self.interrupt_event[websocket].is_set():
                        if self.debug:
                            self.console.print(f"Sending sentence: {complete_sentence}", style="blue")

                        # Send text response
                        await websocket.send(json.dumps({
                            'type': 'response',
                            'text': complete_sentence
                        }))

                        # Generate and send TTS
                        await self._generate_and_send_tts(websocket, complete_sentence)
                        await asyncio.sleep(0.3)

            # Handle remaining text
            if current_sentence.strip() and not self.interrupt_event[websocket].is_set():
                full_response += current_sentence + " "
                await websocket.send(json.dumps({
                    'type': 'response',
                    'text': current_sentence
                }))
                await self._generate_and_send_tts(websocket, current_sentence)

            # Store assistant's response in conversation history
            if full_response.strip():
                self.conversations[websocket].append({
                    "role": "assistant",
                    "content": full_response.strip()
                })

        except asyncio.CancelledError:
            if self.debug:
                self.console.print("Response generation cancelled", style="yellow")
        except Exception as e:
            if self.debug:
                self.console.print(f"Error generating response: {e}", style="bold red")
    async def _generate_and_send_tts(self, websocket, text):
        """Generate TTS and save to temp file, then send file path to client"""
        try:
            if self.interrupt_event[websocket].is_set():
                return

            self.is_speaking[websocket] = True
            
            # Generate unique filename
            filename = f"tts_{time.time_ns()}.wav"
            filepath = os.path.join(self.tts_temp_dir, filename)
            
            # Generate TTS
            wav = await asyncio.to_thread(
                self.tts_model.tts,
                text,
                speaker_idx=0,
                speed=1.6
            )
            
            # Check for interruption after generation
            if self.interrupt_event[websocket].is_set():
                return

            # Save audio file
            audio_data = (np.array(wav) * 32767).astype(np.int16)
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.tts_model.synthesizer.output_sample_rate)
                wf.writeframes(audio_data.tobytes())
            
            # Calculate audio duration
            audio_duration = len(wav) / self.tts_model.synthesizer.output_sample_rate
            
            # Send file path and duration to client
            await websocket.send(json.dumps({
                'type': 'tts_file',
                'filepath': filepath,
                'duration': audio_duration,
                'text': text,
                'complete_sentence': True
            }))

            if self.debug:
                self.console.print(f"Sent TTS file path for: {text}", style="green")

            # Wait for playback duration unless interrupted
            try:
                await asyncio.sleep(audio_duration)
            except asyncio.CancelledError:
                if self.debug:
                    self.console.print("TTS playback interrupted", style="yellow")
                raise

        except asyncio.CancelledError:
            if self.interrupt_event[websocket].is_set():
                await websocket.send(json.dumps({
                    'type': 'interrupt'
                }))
            raise
        except Exception as e:
            if self.debug:
                self.console.print(f"TTS error: {e}", style="bold red")
        finally:
            self.is_speaking[websocket] = False
            # Cleanup file after sending
            try:
                if os.path.exists(filepath):
                    os.unlink(filepath)
                    if self.debug:
                        self.console.print(f"Cleaned up TTS file: {filepath}", style="dim blue")
            except Exception as e:
                if self.debug:
                    self.console.print(f"Error cleaning up TTS file: {e}", style="dim red")

    def _create_prompt(self, websocket):
        """Create prompt using actual conversation history"""
        if websocket not in self.conversations:
            self.conversations[websocket] = [
                {
                    "role": "system",
                    "content": (
                        "You are a professional HR representative from Thaagam Foundation conducting a voice interview "
                        "for a Voice Process position. Keep your responses concise, professional, and focused on "
                        "evaluating the candidate's skills and experience. Ask one question at a time and wait for "
                        "the candidate's response. Do not make up or assume the candidate's responses. Respond naturally "
                        "to what they actually say."
                    )
                }
            ]
        
        conversation = self.conversations[websocket]
        
        # Convert conversation history to prompt format
        prompt = (
            "You are a professional HR representative from Thaagam Foundation conducting a voice interview. "
            "Keep responses concise and professional. Only respond to what the candidate actually says.\n\n"
        )
        
        # Add conversation history
        for msg in conversation[1:]:  # Skip system message
            role = "Candidate" if msg["role"] == "user" else "HR"
            prompt += f"{role}: {msg['content']}\n"
        
        prompt += "HR:"
        return prompt

    async def handle_client(self, websocket):
        try:
            self.active_connections.add(websocket)
            self._init_client_state(websocket)
            self.console.print(f"Client connected. Total clients: {len(self.active_connections)}")

            await websocket.send(json.dumps({
                'type': 'status',
                'message': 'Connected to server'
            }))

            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data['type'] == 'audio':
                        audio_bytes = base64.b64decode(data['data'])
                        await self._process_audio_chunk(websocket, audio_bytes)
                except json.JSONDecodeError:
                    if self.debug:
                        self.console.print("Received invalid JSON")
                except Exception as e:
                    if self.debug:
                        self.console.print(f"Error handling message: {e}", style="bold red")

        except websockets.exceptions.ConnectionClosed:
            self.console.print("Client connection closed")
        finally:
            self._cleanup_client(websocket)
            self.console.print(f"Client disconnected. Total clients: {len(self.active_connections)}")
    async def start_server(self, host: str = 'localhost', port: int = 8765):
        """Start the WebSocket server"""
        self.console.print(f"Starting WebSocket server on ws://{host}:{port}", style="bold blue")
        server = await websockets.serve(
            self.handle_client, 
            host, 
            port,
            max_size=10_485_760  # 10MB max message size
        )
        self.console.print("Server is running. Press Ctrl+C to stop.", style="bold green")
        await server.wait_closed()

def main():
    server = VoiceAssistantServer()
    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()



