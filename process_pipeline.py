# process_pipeline.py - Audio Processing Pipeline
import os
import tempfile
import json
import shutil
from pathlib import Path
from datetime import datetime
from pydub import AudioSegment
from openai import OpenAI
from dotenv import load_dotenv
import logging
import time
import random
from tenacity import retry, stop_after_attempt, wait_exponential

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UrduTranscriptionPipeline:
    def __init__(self):
        # Load API Key
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file")
        
        self.client = OpenAI(api_key=self.api_key)
        
        # Setup folder structure
        self.setup_folders()
        
    def setup_folders(self):
        """Create organized folder structure"""
        self.base_folder = Path("processed_data")
        self.audio_folder = self.base_folder / "audio"
        self.urdu_folder = self.base_folder / "urdu"
        self.english_folder = self.base_folder / "english"
        
        # Create folders if they don't exist
        for folder in [self.base_folder, self.audio_folder, self.urdu_folder, self.english_folder]:
            folder.mkdir(parents=True, exist_ok=True)
        
        logger.info("Folder structure created successfully")
    
    def chunk_audio(self, audio_path, chunk_size_ms=30000, overlap_ms=5000):
        """Split audio into chunks with metadata"""
        logger.info(f"Loading audio file: {audio_path}")
        try:
            audio = AudioSegment.from_file(audio_path)
            duration_ms = len(audio)
            step_ms = chunk_size_ms - overlap_ms
            
            chunks = []
            chunk_metadata = []
            
            for start_ms in range(0, duration_ms, step_ms):
                end_ms = min(start_ms + chunk_size_ms, duration_ms)
                chunk = audio[start_ms:end_ms]
                chunks.append(chunk)
                
                # Add metadata
                chunk_metadata.append({
                    'start_time': self.ms_to_timestamp(start_ms),
                    'end_time': self.ms_to_timestamp(end_ms),
                    'duration_ms': end_ms - start_ms
                })
            
            logger.info(f"Created {len(chunks)} chunks from {duration_ms/1000:.1f}s audio")
            return chunks, chunk_metadata, duration_ms
            
        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
            raise
    
    def ms_to_timestamp(self, ms):
        """Convert milliseconds to timestamp format"""
        minutes = ms // 60000
        seconds = (ms % 60000) // 1000
        milliseconds = (ms % 1000) // 10
        return f"{minutes:02d}:{seconds:02d}:{milliseconds:02d}"
    
    def process_chunks(self, chunks, chunk_metadata):
        """Process each chunk for both transcription and translation - PROVEN APPROACH"""
        results = []
        
        for i, (chunk, metadata) in enumerate(zip(chunks, chunk_metadata)):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}...")
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_path = temp_file.name
            temp_file.close()
            
            try:
                # Export chunk to temporary file
                chunk.export(temp_path, format="mp3")
                
                with open(temp_path, 'rb') as audio_file:
                    # Get Urdu transcription - SIMPLE BASELINE APPROACH
                    audio_file.seek(0)
                    transcription = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language="ur"  # That's it! No prompts, no extra parameters
                    )
                    
                    # Get English translation
                    audio_file.seek(0)
                    translation = self.client.audio.translations.create(
                        model="whisper-1",
                        file=audio_file
                    )
                
                # Process results
                urdu_text = transcription if isinstance(transcription, str) else transcription.text
                english_text = translation if isinstance(translation, str) else translation.text
                
                # Check for Urdu script (for logging)
                import re
                has_urdu_chars = bool(re.search(r'[\u0600-\u06FF\u0750-\u077F]', urdu_text))
                
                result = {
                    'chunk_id': i+1,
                    'start_time': metadata['start_time'],
                    'end_time': metadata['end_time'],
                    'urdu_text': urdu_text,
                    'english_translation': english_text,
                    'urdu_word_count': len(urdu_text.split()),
                    'english_word_count': len(english_text.split()),
                    'has_urdu_script': has_urdu_chars
                }
                results.append(result)
                
                logger.info(f"Chunk {i+1} completed successfully - Urdu script: {has_urdu_chars}")
                
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {e}")
                result = {
                    'chunk_id': i+1,
                    'start_time': metadata['start_time'],
                    'end_time': metadata['end_time'],
                    'urdu_text': "[Error in Urdu transcription]",
                    'english_translation': "[Error in English translation]",
                    'error': str(e),
                    'has_urdu_script': False
                }
                results.append(result)
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        return results

    def process_chunk_with_retries(self, temp_path, urdu_prompt, max_retries=3):
        """Process a single chunk with manual retry logic"""
        import re
        
        for attempt in range(max_retries):
            try:
                with open(temp_path, 'rb') as audio_file:
                    # Get Urdu transcription
                    audio_file.seek(0)
                    logger.info(f"Attempt {attempt + 1}: Getting Urdu transcription...")
                    transcription = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language="ur",
                        prompt=urdu_prompt,
                        response_format="text",
                        temperature=0.0
                    )
                    
                    # Get English translation
                    audio_file.seek(0)
                    logger.info(f"Attempt {attempt + 1}: Getting English translation...")
                    translation = self.client.audio.translations.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="text",
                        temperature=0.0
                    )
                
                # Process results
                urdu_text = transcription if isinstance(transcription, str) else transcription.text
                english_text = translation if isinstance(translation, str) else translation.text
                
                # Check for Urdu script
                has_urdu_chars = bool(re.search(r'[\u0600-\u06FF\u0750-\u077F]', urdu_text))
                
                return urdu_text, english_text, has_urdu_chars
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(1, 3)  # Exponential backoff
                    logger.info(f"Waiting {wait_time:.1f}s before retry...")
                    time.sleep(wait_time)
                else:
                    raise e  # Re-raise on final attempt
        
    def save_processed_data(self, audio_path, results, duration_ms):
        """Save all processed data in organized structure"""
        file_stem = Path(audio_path).stem
        
        # Copy audio file to processed audio folder
        audio_output = self.audio_folder / f"{file_stem}.mp3"
        
        # Convert original audio to MP3 if needed
        audio = AudioSegment.from_file(audio_path)
        audio.export(audio_output, format="mp3")
        logger.info(f"Audio saved to: {audio_output}")
        
        # Combine all transcriptions
        urdu_text = " ".join([r['urdu_text'] for r in results if not r['urdu_text'].startswith('[Error')])
        english_text = " ".join([r['english_translation'] for r in results if not r['english_translation'].startswith('[Error')])
        
        # Save Urdu transcription
        urdu_output = self.urdu_folder / f"{file_stem}.txt"
        with open(urdu_output, "w", encoding="utf-8") as f:
            f.write(urdu_text)
        logger.info(f"Urdu transcription saved to: {urdu_output}")
        
        # Save English translation
        english_output = self.english_folder / f"{file_stem}.txt"
        with open(english_output, "w", encoding="utf-8") as f:
            f.write(english_text)
        logger.info(f"English translation saved to: {english_output}")
        
        # Create detailed JSON for this file
        file_data = {
            "metadata": {
                "original_file": str(audio_path),
                "processed_file": str(audio_output),
                "processing_date": datetime.now().isoformat(),
                "duration_seconds": duration_ms / 1000,
                "total_chunks": len(results)
            },
            "chunks": results,
            "summary": {
                "total_urdu_words": sum(r.get('urdu_word_count', 0) for r in results),
                "total_english_words": sum(r.get('english_word_count', 0) for r in results),
                "successful_chunks": len([r for r in results if not r['urdu_text'].startswith('[Error')])
            }
        }
        
        # Save detailed JSON
        json_output = self.base_folder / f"{file_stem}_detailed.json"
        with open(json_output, "w", encoding="utf-8") as f:
            json.dump(file_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Detailed data saved to: {json_output}")
        
        return file_data
    
    def update_metadata(self, file_data):
        """Update global metadata file"""
        metadata_file = self.base_folder / "metadata.json"
        
        # Load existing metadata
        if metadata_file.exists() and metadata_file.stat().st_size > 0:
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Invalid metadata.json file, creating new one: {e}")
                metadata = {
                    "total_files": 0,
                    "total_duration": 0,
                    "processing_history": [],
                    "avg_accuracy": 95,
                    "last_processed": None
                }
        else:
            metadata = {
                "total_files": 0,
                "total_duration": 0,
                "processing_history": [],
                "avg_accuracy": 95,
                "last_processed": None
            }
        
        # Update metadata
        metadata["total_files"] += 1
        metadata["total_duration"] += file_data["metadata"]["duration_seconds"]
        metadata["last_processed"] = datetime.now().isoformat()
        
        # Add to processing history
        metadata["processing_history"].append({
            "date": datetime.now().isoformat(),
            "filename": Path(file_data["metadata"]["original_file"]).name,
            "duration": file_data["metadata"]["duration_seconds"],
            "chunks": file_data["metadata"]["total_chunks"]
        })
        
        # Save updated metadata
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info("Metadata updated successfully")
    
    def process_file(self, audio_path):
        """Process a single audio file through the complete pipeline"""
        start_time = datetime.now()
        logger.info(f"Starting processing for: {audio_path}")
        
        try:
            # Step 1: Chunk audio
            chunks, chunk_metadata, duration_ms = self.chunk_audio(audio_path)
            
            # Step 2: Process chunks
            results = self.process_chunks(chunks, chunk_metadata)
            
            # Step 3: Save processed data
            file_data = self.save_processed_data(audio_path, results, duration_ms)
            
            # Step 4: Update metadata
            self.update_metadata(file_data)
            
            processing_time = datetime.now() - start_time
            logger.info(f"Processing completed in {processing_time}")
            logger.info(f"Results saved in organized folder structure under: {self.base_folder}")
            
            return file_data
            
        except Exception as e:
            logger.error(f"Failed to process {audio_path}: {e}")
            raise
    
    def process_dataset_folder(self, dataset_path):
        """Process all audio files in a dataset folder"""
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")
        
        # Find all audio files
        audio_extensions = ['.m4a', '.mp3', '.wav', '.flac', '.aac']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(dataset_path.rglob(f"*{ext}"))
        
        if not audio_files:
            logger.warning(f"No audio files found in {dataset_path}")
            return
        
        logger.info(f"Found {len(audio_files)} audio files to process")
        
        # Process each file
        for i, audio_file in enumerate(audio_files, 1):
            logger.info(f"Processing file {i}/{len(audio_files)}: {audio_file.name}")
            try:
                self.process_file(audio_file)
            except Exception as e:
                logger.error(f"Failed to process {audio_file}: {e}")
                continue
        
        logger.info("Dataset processing completed!")

def main():
    """Main execution function"""
    print("🎵 Urdu Audio Transcription Pipeline")
    print("=" * 50)
    
    # Initialize pipeline
    try:
        pipeline = UrduTranscriptionPipeline()
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        return
    
    # Choose processing mode
    print("\n📋 Processing Options:")
    print("1. Process single audio file")
    print("2. Process entire dataset folder")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        # Single file processing
        audio_path = input("Enter path to audio file: ").strip()
        if not os.path.exists(audio_path):
            print(f"❌ File not found: {audio_path}")
            return
        
        try:
            pipeline.process_file(audio_path)
            print(f"✅ Successfully processed: {audio_path}")
        except Exception as e:
            print(f"❌ Error processing file: {e}")
    
    elif choice == "2":
        # Dataset folder processing
        dataset_path = input("Enter path to dataset folder: ").strip()
        if not dataset_path:
            dataset_path = "Dataset"  # Default path
        
        try:
            pipeline.process_dataset_folder(dataset_path)
            print("✅ Dataset processing completed!")
        except Exception as e:
            print(f"❌ Error processing dataset: {e}")
    
    else:
        print("❌ Invalid choice")
        return
    
    print(f"\n📁 Processed files are organized in: {pipeline.base_folder}")
    print("🚀 You can now run the Streamlit app:")
    print("streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()