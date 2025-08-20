import asyncio
import websockets
import json
import base64
import numpy as np
import webrtcvad
import wave
import os
import tempfile
from openai import AsyncOpenAI
from collections import deque
import time
from rich.console import Console
import re
import torch


class VoiceAssistantServer:
    def __init__(self):
        self.console = Console()
        self.console.print("Initializing Voice Assistant Server...", style="bold blue")

        # Audio settings
        self.RATE = 16000
        self.CHANNELS = 1

        # Initialize components
        self.active_connections = set()
        self.client_buffers = {}
        self.client_states = {}
        self.debug = True
        self.conversations = {}
        self.is_speaking = {}
        self.interrupt_event = {}

        # Load models
        self._load_models()

        # FIXED: Use environment variable or set your real API key here
        api_key = "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"
        if not api_key:
            self.console.print("⚠️  OPENAI_API_KEY not found in environment", style="bold yellow")
            self.console.print("Set it with: export OPENAI_API_KEY='sk-your-real-key-here'", style="yellow")
            # For testing, you can uncomment and add your real key:
            # api_key = "sk-your-real-openai-api-key-here"

        try:
            self.openai_client = AsyncOpenAI(api_key=api_key) if api_key else None
        except Exception as e:
            self.console.print(f"❌ OpenAI client initialization failed: {e}", style="red")
            self.openai_client = None

        # LLM Client
        self.llm_client = AsyncOpenAI(
            api_key='ollama',
            base_url='http://localhost:11434/v1'
        )

    def _load_models(self):
        try:
            # Load TTS
            from TTS.api import TTS
            use_gpu = torch.cuda.is_available()
            self.tts_model = TTS(
                model_name="tts_models/en/ljspeech/vits",
                progress_bar=False,
                gpu=use_gpu
            )

            # Load VAD
            self.vad = webrtcvad.Vad(2)  # Less aggressive

            self.console.print("✅ Models loaded successfully!", style="bold green")
        except Exception as e:
            self.console.print(f"❌ Error loading models: {e}", style="bold red")
            raise

        self.tts_temp_dir = tempfile.mkdtemp(prefix='tts_audio_')

    def _prepare_frame(self, audio_data):
        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            frame_size = int(self.RATE * 0.03)  # 30ms

            if len(audio_np) < frame_size:
                audio_np = np.pad(audio_np, (0, frame_size - len(audio_np)))
            elif len(audio_np) > frame_size:
                audio_np = audio_np[:frame_size]

            return audio_np.tobytes()
        except Exception as e:
            return None

    async def _process_audio_chunk(self, websocket, audio_data):
        """Improved audio processing with longer speech collection"""
        try:
            state = self.client_states[websocket]
            buffers = self.client_buffers[websocket]

            # Skip if AI is speaking
            if self.is_speaking.get(websocket, False):
                return

            frame = self._prepare_frame(audio_data)
            if not frame:
                return

            # Enhanced energy calculation with noise gate
            try:
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                if len(audio_np) == 0:
                    return

                # Apply simple noise gate
                audio_float = audio_np.astype(np.float32)
                energy = float(np.sqrt(np.mean(audio_float ** 2)))

                # Higher quality threshold
                if np.isnan(energy) or energy < 300:  # Reduced but still filtered
                    return

            except:
                return

            is_speech = self.vad.is_speech(frame, self.RATE)

            if is_speech and energy > 600:  # Moderate energy threshold
                state['silence_frames'] = 0
                if not state.get('speech_detected', False):
                    state['speech_detected'] = True
                    state['speech_start_time'] = time.time()  # Track speech start
                    self.console.print(f"🎤 Speech started (energy: {energy:.0f})", style="cyan")
                buffers['speech_buffer'].append(audio_data)

            else:
                if state.get('speech_detected', False):
                    state['silence_frames'] = state.get('silence_frames', 0) + 1
                    buffers['speech_buffer'].append(audio_data)

                    # CRITICAL FIX: Wait for longer silence (1.5 seconds) to ensure complete speech
                    if state['silence_frames'] > 50:  # ~1.5 seconds of silence
                        speech_duration = time.time() - state.get('speech_start_time', 0)

                        # CRITICAL FIX: Require minimum 2 seconds of actual speech
                        if len(buffers['speech_buffer']) > 60 and speech_duration > 2.0:
                            speech_data = b''.join(buffers['speech_buffer'])

                            # Validate audio length
                            audio_duration_seconds = len(speech_data) / (self.RATE * 2)  # 16-bit = 2 bytes

                            if audio_duration_seconds >= 2.0:  # Minimum 2 seconds
                                buffers['speech_buffer'] = []
                                state['speech_detected'] = False
                                state['silence_frames'] = 0

                                self.console.print(f"🔄 Processing {audio_duration_seconds:.1f}s of speech",
                                                   style="bold blue")
                                await self._process_speech_enhanced(websocket, speech_data)
                            else:
                                self.console.print(f"🚫 Speech too short: {audio_duration_seconds:.1f}s", style="yellow")
                                self._reset_speech_state(websocket)
                        else:
                            self.console.print(f"🚫 Insufficient speech data: {len(buffers['speech_buffer'])} chunks",
                                               style="yellow")
                            self._reset_speech_state(websocket)

        except Exception as e:
            self.console.print(f"❌ Audio processing error: {e}", style="red")

    async def _process_speech_enhanced(self, websocket, audio_data):
        """Enhanced speech processing with better OpenAI parameters"""
        temp_file = None
        try:
            self.console.print("🚀 ENHANCED SPEECH PROCESSING", style="bold green")

            # Validate audio data thoroughly
            if len(audio_data) < 32000:  # Minimum ~2 seconds at 16kHz
                self.console.print("❌ Audio data insufficient for quality transcription", style="red")
                return False

            audio_duration = len(audio_data) / (self.RATE * 2)
            self.console.print(f"📊 Processing {audio_duration:.1f}s of audio ({len(audio_data)} bytes)", style="blue")

            # Create high-quality temp file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name

            with wave.open(temp_file, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(self.RATE)
                wf.writeframes(audio_data)

            file_size = os.path.getsize(temp_file)
            self.console.print(f"✅ Created audio file: {file_size} bytes", style="green")

            if not self.openai_client:
                self.console.print("❌ OpenAI client not available", style="red")
                return False

            # CRITICAL FIX: Enhanced OpenAI API call with optimal parameters
            try:
                with open(temp_file, "rb") as audio_file:
                    response = await self.openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="text",
                        language="en",  # CRITICAL: Specify language for better accuracy
                        prompt="This is a professional HR interview conversation about customer service experience and job qualifications.",
                        # Context helps accuracy
                        temperature=0.0  # Lower temperature for more focused transcription
                    )

                text = response.strip() if isinstance(response, str) else str(response).strip()
                self.console.print("✅ OpenAI transcription completed", style="green")

            except Exception as api_error:
                self.console.print(f"❌ OpenAI API error: {api_error}", style="bold red")
                return False

            # Enhanced validation with better filtering
            if not text or len(text.strip()) < 5:  # Require at least 5 characters
                self.console.print(f"🚫 Transcription too short: '{text}'", style="yellow")
                return False

            # Improved false positive detection
            text_clean = text.lower().strip()
            false_positives = ['hello', 'hello?', 'thank you', 'mm-hmm', 'uh', 'um']

            if text_clean in false_positives or len(text_clean.split()) < 3:  # Require at least 3 words
                self.console.print(f"🚫 Filtered insufficient content: '{text}'", style="yellow")
                return False

            # SUCCESS - Valid transcription
            self.console.print(f"✅ COMPLETE TRANSCRIPTION: '{text}'", style="bold green")

            # Store in conversation
            if websocket not in self.conversations:
                self._init_client_state(websocket)

            self.conversations[websocket].append({
                "role": "user",
                "content": text
            })

            # Send to client
            await websocket.send(json.dumps({
                'type': 'transcription',
                'text': text,
                'duration': audio_duration
            }))

            # Generate contextual response
            await self._generate_contextual_response(websocket, text)

            return True

        except Exception as e:
            self.console.print(f"❌ Enhanced speech processing error: {e}", style="bold red")
            import traceback
            traceback.print_exc()
            return False
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass

    async def _generate_contextual_response(self, websocket, user_text):
        """Generate smarter contextual responses"""
        try:
            # Analyze user input for better responses
            user_lower = user_text.lower()

            # More sophisticated response logic
            if any(word in user_lower for word in ['experience', 'years', 'worked']):
                response_text = "That's excellent experience! Can you describe a specific challenging situation you handled successfully?"
            elif any(word in user_lower for word in ['customer', 'service', 'help']):
                response_text = "Great! What do you think is the most important quality for excellent customer service?"
            elif any(word in user_lower for word in ['team', 'colleagues', 'work']):
                response_text = "Teamwork is crucial! How do you handle conflicts or disagreements with team members?"
            elif any(word in user_lower for word in ['skills', 'abilities', 'good']):
                response_text = "Wonderful! Can you give me a specific example of how you used those skills?"
            else:
                response_text = "That sounds very interesting! Could you provide more specific details about that experience?"

            self.console.print(f"🤖 Contextual response: {response_text}", style="blue")

            await websocket.send(json.dumps({
                'type': 'response',
                'text': response_text
            }))

            # Generate TTS
            await self._generate_simple_tts(websocket, response_text)

        except Exception as e:
            self.console.print(f"❌ Response generation error: {e}", style="red")

    def _reset_speech_state(self, websocket):
        """Reset speech detection state"""
        state = self.client_states[websocket]
        buffers = self.client_buffers[websocket]

        buffers['speech_buffer'] = []
        state['speech_detected'] = False
        state['silence_frames'] = 0
        if 'speech_start_time' in state:
            del state['speech_start_time']

    async def _process_speech_immediate(self, websocket, audio_data):
        """Process speech immediately without task scheduling"""
        temp_file = None
        try:
            self.console.print("🚀 SPEECH PROCESSING STARTING", style="bold green")

            if len(audio_data) < 1000:
                self.console.print("❌ Audio too small", style="red")
                return False

            # Create temp file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name

            with wave.open(temp_file, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(self.RATE)
                wf.writeframes(audio_data)

            self.console.print(f"📁 Created audio file: {os.path.getsize(temp_file)} bytes", style="blue")

            # Check OpenAI client
            if not self.openai_client:
                self.console.print("❌ OpenAI client not available - using mock transcription", style="red")
                # Mock transcription for testing
                text = "I have 3 years of customer service experience"
            else:
                # Real OpenAI transcription
                try:
                    with open(temp_file, "rb") as audio_file:
                        response = await self.openai_client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file,
                            response_format="text"
                        )

                    text = response.strip() if isinstance(response, str) else str(response).strip()
                    self.console.print("✅ OpenAI transcription successful", style="green")

                except Exception as api_error:
                    self.console.print(f"❌ OpenAI API error: {api_error}", style="red")
                    return False

            # Validate transcription
            if not text or len(text.strip()) < 3:
                self.console.print("🚫 Transcription too short", style="yellow")
                return False

            # Filter obvious false positives
            text_lower = text.lower().strip()
            if text_lower in ['hello', 'thank you', 'mm-hmm', 'you']:
                self.console.print(f"🚫 Filtered: {text}", style="yellow")
                return False

            # SUCCESS!
            self.console.print(f"✅ VALID TRANSCRIPTION: '{text}'", style="bold green")

            # Store conversation
            if websocket not in self.conversations:
                self._init_client_state(websocket)

            self.conversations[websocket].append({
                "role": "user",
                "content": text
            })

            # Send to client
            await websocket.send(json.dumps({
                'type': 'transcription',
                'text': text
            }))

            # Generate response
            await self._generate_simple_response(websocket, text)

            return True

        except Exception as e:
            self.console.print(f"❌ Speech processing error: {e}", style="bold red")
            import traceback
            traceback.print_exc()
            return False
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass

    async def _generate_simple_response(self, websocket, user_text):
        """Generate a simple response"""
        try:
            # Simple rule-based responses for testing
            responses = {
                "experience": "That's great! Can you tell me about a challenging customer situation you handled?",
                "customer": "Excellent! What was your biggest achievement in customer service?",
                "service": "Perfect! How do you handle difficult customers?",
                "default": "That sounds interesting! Can you elaborate on that experience?"
            }

            # Choose response based on keywords
            user_lower = user_text.lower()
            response_text = responses["default"]

            for keyword, response in responses.items():
                if keyword in user_lower:
                    response_text = response
                    break

            self.console.print(f"🤖 AI Response: {response_text}", style="blue")

            # Send response
            await websocket.send(json.dumps({
                'type': 'response',
                'text': response_text
            }))

            # Generate TTS
            await self._generate_simple_tts(websocket, response_text)

        except Exception as e:
            self.console.print(f"❌ Response generation error: {e}", style="red")

    async def _generate_simple_tts(self, websocket, text):
        """Generate TTS without complex state management"""
        try:
            self.is_speaking[websocket] = True

            filename = f"tts_{time.time_ns()}.wav"
            filepath = os.path.join(self.tts_temp_dir, filename)

            # Generate TTS
            wav = await asyncio.to_thread(self.tts_model.tts, text)

            # Save file
            audio_data = (np.array(wav) * 32767).astype(np.int16)
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(22050)  # TTS model sample rate
                wf.writeframes(audio_data.tobytes())

            duration = len(wav) / 22050

            await websocket.send(json.dumps({
                'type': 'tts_file',
                'filepath': filepath,
                'duration': duration,
                'text': text
            }))

            self.console.print(f"🔊 TTS sent: {text}", style="green")

            # Wait for TTS to complete
            await asyncio.sleep(duration + 0.5)

        except Exception as e:
            self.console.print(f"❌ TTS error: {e}", style="red")
        finally:
            self.is_speaking[websocket] = False

    def _init_client_state(self, websocket):
        self.client_buffers[websocket] = {
            'speech_buffer': [],
        }
        self.client_states[websocket] = {
            'silence_frames': 0,
            'speech_detected': False,
        }
        self.is_speaking[websocket] = False
        self.interrupt_event[websocket] = asyncio.Event()

        # Simple conversation
        self.conversations[websocket] = [
            {
                "role": "assistant",
                "content": "Hello! I'm calling from Thaagam Foundation about your Voice Process application. Could you tell me about your customer service experience?"
            }
        ]

    async def handle_client(self, websocket):
        try:
            self.active_connections.add(websocket)
            self._init_client_state(websocket)
            self.console.print(f"✅ Client connected. Total: {len(self.active_connections)}")

            # Send initial greeting
            await asyncio.sleep(1)
            initial_message = self.conversations[websocket][0]['content']
            await self._generate_simple_tts(websocket, initial_message)

            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data['type'] == 'audio':
                        audio_bytes = base64.b64decode(data['data'])
                        await self._process_audio_chunk(websocket, audio_bytes)
                except:
                    pass

        except websockets.exceptions.ConnectionClosed:
            self.console.print("Client disconnected")
        finally:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def start_server(self, host='localhost', port=8765):
        self.console.print(f"🚀 Starting server on ws://{host}:{port}", style="bold blue")
        server = await websockets.serve(self.handle_client, host, port)
        await server.wait_closed()


def main():
    server = VoiceAssistantServer()
    asyncio.run(server.start_server())


if __name__ == "__main__":
    main()
