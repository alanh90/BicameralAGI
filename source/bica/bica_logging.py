import logging
from logging.handlers import RotatingFileHandler
import os
import traceback
from datetime import datetime


class BicaLogging:
    def __init__(self, name, log_file='bica.log', log_level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)

        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        c_handler = logging.StreamHandler()
        f_handler = RotatingFileHandler(os.path.join(log_dir, log_file), maxBytes=10 * 1024 * 1024, backupCount=5)

        log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(log_format)
        f_handler.setFormatter(log_format)

        self.logger.addHandler(c_handler)
        self.logger.addHandler(f_handler)

    def _log(self, level, message, **kwargs):
        extra = ', '.join(f"{k}={v}" for k, v in kwargs.items() if v is not None)
        log_msg = f"{message} | {extra}" if extra else message
        getattr(self.logger, level)(log_msg)

    def debug(self, message, **kwargs):
        self._log('debug', message, **kwargs)

    def info(self, message, **kwargs):
        self._log('info', message, **kwargs)

    def warning(self, message, **kwargs):
        self._log('warning', message, **kwargs)

    def error(self, message, exc_info=False, **kwargs):
        self._log('error', message, exc_info=exc_info, **kwargs)
        if exc_info:
            self.logger.error(traceback.format_exc())

    def critical(self, message, notify_admin=False, **kwargs):
        self._log('critical', message, **kwargs)
        if notify_admin:
            # Placeholder for admin notification logic
            print(f"ADMIN NOTIFICATION: {message}")

    def log_chamber_interaction(self, from_chamber, to_chamber, message):
        self.info(f"Chamber Interaction: {from_chamber} -> {to_chamber}", message=message)

    def log_emotion_change(self, emotion, intensity, trigger=None):
        self.info(f"Emotion Change: {emotion}", intensity=intensity, trigger=trigger)

    def log_memory_operation(self, operation, memory_id, content=None, importance=None):
        self.info(f"Memory {operation}", id=memory_id, content=content, importance=importance)

    def log_thought_process(self, thought, category=None, influences=None):
        self.debug(f"Thought Process: {thought}", category=category, influences=influences)

    def log_action_execution(self, action, result, context=None):
        self.info(f"Action Executed: {action}", result=result, context=context)

    def log_personality_update(self, trait, old_value, new_value, reason=None):
        self.info(f"Personality Update: {trait}", old=old_value, new=new_value, reason=reason)

    def log_safety_check(self, check_type, result, details=None):
        self.info(f"Safety Check: {check_type}", result=result, details=details)

    def log_error_boundary(self, error_type, message, module=None):
        self.error(f"Error Boundary: {error_type}", message=message, module=module)


# Example usage
if __name__ == "__main__":
    logger = BicaLogging("BicaTest")
    logger.log_chamber_interaction("ThoughtChamber", "ActionChamber", "Initiating action based on thought")
    logger.log_emotion_change("joy", 0.8, trigger="User compliment")
    logger.log_memory_operation("store", "mem001", content="First conversation with user", importance=0.7)
    logger.log_thought_process("I should ask about the user's day", category="social", influences=["empathy", "curiosity"])
    logger.log_action_execution("ask_about_day", "User responded positively", context="Casual conversation")
    logger.log_personality_update("openness", 0.6, 0.7, reason="Positive interaction with new topic")
    logger.log_safety_check("content_filter", "pass", details="No unsafe content detected")
    logger.log_error_boundary("RuntimeError", "Unexpected response from external API", module="WeatherService")
