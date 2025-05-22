# streamlit_app.py - Main Streamlit Application for Deployment
import streamlit as st
import os
import json
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64

# Set page config
st.set_page_config(
    page_title="Urdu Audio Transcription Pipeline",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .transcription-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    .urdu-text {
        direction: rtl;
        text-align: right;
        font-size: 16px;
        line-height: 1.6;
    }
    .english-text {
        font-size: 16px;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

def load_processed_data():
    """Load all processed transcription data"""
    data_folder = Path("processed_data")
    
    if not data_folder.exists():
        return None, None, None
    
    # Load metadata
    metadata_file = data_folder / "metadata.json"
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
    # Get all audio files
    audio_folder = data_folder / "audio"
    audio_files = list(audio_folder.glob("*.mp3")) if audio_folder.exists() else []
    
    return metadata, audio_files, data_folder

def get_transcription_files(data_folder, audio_file_stem):
    """Get corresponding transcription files for an audio file"""
    urdu_file = data_folder / "urdu" / f"{audio_file_stem}.txt"
    english_file = data_folder / "english" / f"{audio_file_stem}.txt"
    
    urdu_text = ""
    english_text = ""
    
    if urdu_file.exists():
        with open(urdu_file, 'r', encoding='utf-8') as f:
            urdu_text = f.read()
    
    if english_file.exists():
        with open(english_file, 'r', encoding='utf-8') as f:
            english_text = f.read()
    
    return urdu_text, english_text

def create_audio_player(audio_file):
    """Create HTML5 audio player"""
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
    
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
    <audio controls style="width: 100%;">
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        Your browser does not support the audio element.
    </audio>
    """
    return audio_html

def create_statistics_dashboard(metadata):
    """Create statistics dashboard"""
    if not metadata:
        st.warning("No processed data found. Please run the processing pipeline first.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Total Files</h3>
            <h2>{}</h2>
        </div>
        """.format(metadata.get('total_files', 0)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Total Duration</h3>
            <h2>{}</h2>
        </div>
        """.format(metadata.get('total_duration', '0:00')), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Avg Accuracy</h3>
            <h2>{}%</h2>
        </div>
        """.format(metadata.get('avg_accuracy', 95)), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>Last Processed</h3>
            <h2>{}</h2>
        </div>
        """.format(metadata.get('last_processed', 'N/A')), unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎵 Urdu Audio Transcription Pipeline</h1>
        <p>Interactive Visualization & Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    metadata, audio_files, data_folder = load_processed_data()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["Dashboard", "Audio Player", "Transcription Viewer", "Analytics"])
    
    if page == "Dashboard":
        st.header("📊 Processing Dashboard")
        create_statistics_dashboard(metadata)
        
        if metadata:
            # Processing timeline
            st.subheader("Processing Timeline")
            if 'processing_history' in metadata:
                df = pd.DataFrame(metadata['processing_history'])
                fig = px.line(df, x='date', y='files_processed', 
                             title="Files Processed Over Time")
                st.plotly_chart(fig, use_container_width=True)
        
        # Instructions
        st.subheader("📋 How to Use")
        st.markdown("""
        1. **Process Audio Files**: Run `python process_pipeline.py` to process your dataset
        2. **View Results**: Use the Audio Player to listen and view transcriptions
        3. **Analyze Data**: Check Analytics for detailed insights
        4. **Export Results**: Download processed data from the Analytics page
        """)
    
    elif page == "Audio Player":
        st.header("🎵 Audio Player & Transcription Viewer")
        
        if not audio_files:
            st.warning("No processed audio files found. Please run the processing pipeline first.")
            st.code("python process_pipeline.py", language="bash")
            return
        
        # File selector
        selected_file = st.selectbox("Select an audio file", 
                                   options=[f.name for f in audio_files])
        
        if selected_file:
            audio_path = None
            for f in audio_files:
                if f.name == selected_file:
                    audio_path = f
                    break
            
            if audio_path:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("🎵 Audio Player")
                    audio_html = create_audio_player(audio_path)
                    st.markdown(audio_html, unsafe_allow_html=True)
                    
                    # File info
                    file_stats = audio_path.stat()
                    st.write(f"**File Size:** {file_stats.st_size / 1024 / 1024:.2f} MB")
                    st.write(f"**File Name:** {audio_path.name}")
                
                with col2:
                    st.subheader("📄 Transcription")
                    
                    # Get transcriptions
                    file_stem = audio_path.stem
                    urdu_text, english_text = get_transcription_files(data_folder, file_stem)
                    
                    # Tabs for languages
                    urdu_tab, english_tab = st.tabs(["اردو متن", "English Text"])
                    
                    with urdu_tab:
                        st.markdown(f"""
                        <div class="transcription-box urdu-text">
                            {urdu_text if urdu_text else "No Urdu transcription available"}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with english_tab:
                        st.markdown(f"""
                        <div class="transcription-box english-text">
                            {english_text if english_text else "No English translation available"}
                        </div>
                        """, unsafe_allow_html=True)
    
    elif page == "Transcription Viewer":
        st.header("📝 Batch Transcription Viewer")
        
        if not audio_files:
            st.warning("No processed files found.")
            return
        
        # Show all transcriptions
        for audio_file in audio_files:
            with st.expander(f"📁 {audio_file.name}"):
                file_stem = audio_file.stem
                urdu_text, english_text = get_transcription_files(data_folder, file_stem)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("اردو")
                    st.markdown(f"""
                    <div class="transcription-box urdu-text">
                        {urdu_text[:500]}{'...' if len(urdu_text) > 500 else ''}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.subheader("English")
                    st.markdown(f"""
                    <div class="transcription-box english-text">
                        {english_text[:500]}{'...' if len(english_text) > 500 else ''}
                    </div>
                    """, unsafe_allow_html=True)
    
    elif page == "Analytics":
        st.header("📈 Analytics & Insights")
        
        if not audio_files:
            st.warning("No processed files found.")
            return
        
        # Word count analysis
        word_counts = []
        for audio_file in audio_files:
            file_stem = audio_file.stem
            urdu_text, english_text = get_transcription_files(data_folder, file_stem)
            
            word_counts.append({
                'filename': audio_file.name,
                'urdu_words': len(urdu_text.split()) if urdu_text else 0,
                'english_words': len(english_text.split()) if english_text else 0,
                'urdu_chars': len(urdu_text) if urdu_text else 0,
                'english_chars': len(english_text) if english_text else 0
            })
        
        df = pd.DataFrame(word_counts)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.bar(df, x='filename', y=['urdu_words', 'english_words'],
                         title="Word Count Comparison", barmode='group')
            fig1.update_xaxis(tickangle=45)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.scatter(df, x='urdu_words', y='english_words',
                             hover_data=['filename'],
                             title="Urdu vs English Word Count Correlation")
            st.plotly_chart(fig2, use_container_width=True)
        
        # Data table
        st.subheader("📊 Detailed Statistics")
        st.dataframe(df, use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Statistics CSV",
            data=csv,
            file_name=f"transcription_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()