"""Voice conditioning column definitions (extracted from vecamp_train/dataset.py)."""

EMOTION_COLUMNS = [
    'Affection', 'Age', 'Amusement', 'Anger', 'Arousal', 'Astonishment_Surprise',
    'Authenticity', 'Awe', 'Background_Noise', 'Bitterness', 'Concentration',
    'Confident_vs._Hesitant', 'Confusion', 'Contemplation', 'Contempt', 'Contentment',
    'Disappointment', 'Disgust', 'Distress', 'Doubt', 'Elation', 'Embarrassment',
    'Emotional_Numbness', 'Fatigue_Exhaustion', 'Fear', 'Gender', 'Helplessness',
    'High_Pitched_vs._Low_Pitched', 'Hope_Enthusiasm_Optimism', 'Impatience_and_Irritability',
    'Infatuation', 'Interest', 'Intoxication_Altered_States_of_Consciousness',
    'Jealousy___Envy', 'Longing', 'Malevolence_Malice', 'Monotone_vs._Expressive',
    'Pain', 'Pleasure_Ecstasy', 'Pride', 'Recording_Quality', 'Relief', 'Sadness',
    'Serious_vs._Humorous', 'Sexual_Lust', 'Shame', 'Soft_vs._Harsh', 'Sourness',
    'Submissive_vs._Dominant', 'Teasing', 'Thankfulness_Gratitude', 'Triumph',
    'Valence', 'Vulnerable_vs._Emotionally_Detached', 'Warm_vs._Cold'
]

METRIC_COLUMNS = ['sisdr', 'pesq', 'stoi', 'snr', 'reverb', 'speaking_rate']
