from bica_logging import BicaLogging

class BicaSafety:
    def __init__(self):
        self.logger = BicaLogging("BicaSafety")

    def check_content(self, content):
        unsafe_keywords = ["violence", "hate", "illegal"]
        for keyword in unsafe_keywords:
            if keyword in content.lower():
                self.logger.warning(f"Potentially unsafe content: {keyword}")
                return False
        return True

    def check_action(self, action):
        unsafe_actions = ["share_personal_info", "make_financial_transaction", "access_system_files"]
        if action in unsafe_actions:
            self.logger.warning(f"Potentially unsafe action: {action}")
            return False
        return True

    def regulate_emotion(self, emotion, intensity):
        if intensity > 0.9:
            self.logger.info(f"Regulating high intensity emotion: {emotion}")
            return 0.9
        return intensity

    def run_safety_check(self, check_type, data):
        if check_type == "content":
            return self.check_content(data)
        elif check_type == "action":
            return self.check_action(data)
        elif check_type == "emotion":
            return self.regulate_emotion(data["emotion"], data["intensity"])
        else:
            self.logger.warning(f"Unknown safety check type: {check_type}")
            return True  # Default to allowing the operation if check type is unknown