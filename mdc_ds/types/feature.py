from datasets import Audio, Features, Value

feature = Features(
    {
        "audio": Audio(sampling_rate=16000),
        "text": Value("string"),
        "language": Value("string"),
        "duration": Value("float32"),
    }
)
