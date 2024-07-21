import requests
import json
import io


class ElevenLabsTTS:
    def __init__(self, api_key):
        self.api_key = api_key
        self.tts_url = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
        self.headers = {
            "Accept": "application/json",
            "xi-api-key": self.api_key
        }

    def generate_speech(self, text, voice_id, latency_optimization=5):
        data = {
            "text": text,
            "model_id": "eleven_turbo_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.8,
                "style": 0.2,
                "use_speaker_boost": True
            },
            "optimize_streaming_latency": latency_optimization
        }

        response = requests.post(self.tts_url.format(voice_id=voice_id), headers=self.headers, json=data, stream=True)

        if response.ok:
            audio_data = io.BytesIO(response.content)
            return audio_data
        else:
            print(response.text)
            return None
