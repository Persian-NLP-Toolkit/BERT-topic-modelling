# from PreprocessPipeline import PreprocessPipeline
class PreprocessPipeline:
    pass

pipeline = PreprocessPipeline()


def safe_preprocess(text):
    try:
        tokens, _ = pipeline.preprocess_pipeline(text)
        return tokens if isinstance(tokens, list) else []
    except Exception as e:
        print(f"⚠️ Error preprocessing: {text[:30]}... | {e}")
        return []
