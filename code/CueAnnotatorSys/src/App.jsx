import { useState, useEffect, useRef, useCallback } from 'react';
import { SAMPLE_INDEX } from './sampleIndex';
import { DATASET_CONFIG, ANNOTATOR_DATASET, STORAGE_KEY } from './config';
import './App.css';

function App() {
  const [currentAnnotator, setCurrentAnnotator] = useState(null);
  const [currentDataset, setCurrentDataset] = useState(null);
  const [currentSample, setCurrentSample] = useState(null);
  const [cueData, setCueData] = useState(null);
  const [annotations, setAnnotations] = useState([]);
  const [completedSamples, setCompletedSamples] = useState({});
  const [audioTime, setAudioTime] = useState(0);
  const [audioDuration, setAudioDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [volume, setVolume] = useState(10);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentSampleIndex, setCurrentSampleIndex] = useState(0);
  const [showAddCueForm, setShowAddCueForm] = useState(false);
  const [newCueText, setNewCueText] = useState('');
  const [newCueStart, setNewCueStart] = useState('');
  const [newCueEnd, setNewCueEnd] = useState('');
  const [originalCues, setOriginalCues] = useState([]);
  const [cuePlaybackEnd, setCuePlaybackEnd] = useState(null);
  const [isAnnotated, setIsAnnotated] = useState(false);
  const [audioKey, setAudioKey] = useState(0);

  const audioRef = useRef(null);
  const audioContextRef = useRef(null);
  const gainNodeRef = useRef(null);
  const sourceNodeRef = useRef(null);
  const mainRef = useRef(null);

  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      setCompletedSamples(JSON.parse(stored));
    }

    // Toggle play/pause with the space bar.
    const handleKeyDown = (e) => {
      if (e.code === 'Space' && !e.repeat && cueData) {
        e.preventDefault();
        togglePlay();
      }
    };

    document.addEventListener('keydown', handleKeyDown);

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      if (audioContextRef.current) {
        audioContextRef.current.close();
        audioContextRef.current = null;
        gainNodeRef.current = null;
        sourceNodeRef.current = null;
      }
    };
  }, [cueData]);

  const selectAnnotator = async (annotator) => {
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
      gainNodeRef.current = null;
      sourceNodeRef.current = null;
    }
    setCurrentAnnotator(annotator);
    const dataset = ANNOTATOR_DATASET[annotator];
    setCurrentDataset(dataset);
    setVolume(10);
    setPlaybackRate(1);

    // Find the first sample without a saved annotation.
    const samples = SAMPLE_INDEX[dataset];
    let firstUnannotatedIndex = 0;
    let found = false;

    for (let i = 0; i < samples.length; i++) {
      const sampleId = samples[i];
      try {
        const response = await fetch(`/api/annotation/${dataset}/${sampleId}`);
        if (!response.ok) {
          firstUnannotatedIndex = i;
          found = true;
          break;
        }
        const result = await response.json();
        if (!result.exists) {
          firstUnannotatedIndex = i;
          found = true;
          break;
        }
      } catch (e) {
        firstUnannotatedIndex = i;
        found = true;
        break;
      }
    }

    if (!found) {
      console.log('All samples have already been annotated.');
    }

    setCurrentSampleIndex(firstUnannotatedIndex);
    await loadSampleAtIndex(dataset, firstUnannotatedIndex);
  };

  const loadSampleAtIndex = useCallback(async (dataset, startIndex) => {
    setIsLoading(true);
    setError(null);
    setIsPlaying(false);
    setAudioTime(0);
    setAudioDuration(0);
    setIsAnnotated(false);
    setAudioKey(prev => prev + 1);
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
      gainNodeRef.current = null;
      sourceNodeRef.current = null;
    }

    const samples = SAMPLE_INDEX[dataset];
    const datasetConfig = DATASET_CONFIG[dataset];

    const loadData = async (index) => {
      if (index >= samples.length) {
        setCurrentSample(null);
        setCueData(null);
        setIsLoading(false);
        setError('All samples in this dataset have been completed.');
        return;
      }

      const sampleId = samples[index];

      let hasAnnotation = false;
      let savedData = null;
      try {
        const annotationResponse = await fetch(`/api/annotation/${dataset}/${sampleId}`);
        const annotationResult = await annotationResponse.json();
        if (annotationResult.exists && annotationResult.data) {
          hasAnnotation = true;
          savedData = annotationResult.data;
        }
      } catch (e) {
        console.log('No saved annotation found');
      }

      setCurrentSampleIndex(index);
      setCurrentSample(sampleId);
      setIsAnnotated(hasAnnotation);

      try {
        let cueJsonPath;
        if (dataset === 'PDCH') {
          const [session, sub] = sampleId.split('_');
          cueJsonPath = `${datasetConfig.basePath}/${session}/${sub}/cue_detection.json`;
        } else if (dataset === 'CMDC') {
          const match = sampleId.match(/(part\d+)(HC\d+|MDD\d+)(Q\d+)/i);
          if (match) {
            const [, part, subject, q] = match;
            cueJsonPath = `${datasetConfig.basePath}/${part}/${subject}/${q}/cue_detection.json`;
          }
        } else {
          cueJsonPath = `${datasetConfig.basePath}/${sampleId}/cue_detection.json`;
        }

        if (!cueJsonPath) {
          setCueData({ cues: [], statistics: {} });
          setAnnotations([]);
          setOriginalCues([]);
          setIsLoading(false);
          return;
        }

        const response = await fetch(cueJsonPath);
        if (!response.ok) {
          setCueData({ cues: [], statistics: {} });
          setAnnotations([]);
          setOriginalCues([]);
          setIsLoading(false);
          return;
        }

        const data = await response.json();

        if (hasAnnotation && savedData) {
          setCueData({ cues: savedData.cues, statistics: {} });
          setOriginalCues(savedData.cues.filter(c => c.status !== 'added').map(cue => ({
            cue_id: cue.cue_id,
            text: cue.text,
            original_span: cue.original_span
          })));
          setAnnotations(savedData.cues.filter(c => c.status !== 'deleted'));
        } else {
          setCueData(data);
          setOriginalCues(data.cues.map((cue, idx) => ({
            cue_id: cue.id ?? idx,
            text: cue.text,
            original_span: { start: cue.start, end: cue.end }
          })));
          setAnnotations(data.cues.map((cue, idx) => ({
            cue_id: cue.id ?? idx,
            text: cue.text,
            original_span: { start: cue.start, end: cue.end },
            corrected_span: { start: cue.start, end: cue.end }
          })));
        }
        setIsLoading(false);
      } catch (err) {
        console.error(`Failed to load ${sampleId}:`, err);
        setCueData({ cues: [], statistics: {} });
        setAnnotations([]);
        setOriginalCues([]);
        setIsLoading(false);
      }
    };

    loadData(startIndex);
  }, []);

  const loadSampleAtIndexStrict = useCallback(async (dataset, targetIndex) => {
    const samples = SAMPLE_INDEX[dataset];
    const datasetConfig = DATASET_CONFIG[dataset];

    if (targetIndex < 0 || targetIndex >= samples.length) {
      return false;
    }

    setIsLoading(true);
    setError(null);
    setIsPlaying(false);
    setAudioTime(0);
    setAudioDuration(0);
    setIsAnnotated(false);
    setAudioKey(prev => prev + 1);

    const sampleId = samples[targetIndex];

    try {
      const annotationResponse = await fetch(`/api/annotation/${dataset}/${sampleId}`);
      const annotationResult = await annotationResponse.json();

      if (annotationResult.exists && annotationResult.data) {
        const savedData = annotationResult.data;
        setCurrentSampleIndex(targetIndex);
        setCurrentSample(sampleId);
        setCueData({ cues: savedData.cues, statistics: {} });
        setOriginalCues(savedData.cues.filter(c => c.status !== 'added').map(cue => ({
          cue_id: cue.cue_id,
          text: cue.text,
          original_span: cue.original_span
        })));
        setAnnotations(savedData.cues.filter(c => c.status !== 'deleted'));
        setIsAnnotated(true);
        setIsLoading(false);
        return true;
      }
    } catch (e) {
      console.log('No saved annotation found, loading original data');
    }

    try {
      let cueJsonPath;
      if (dataset === 'PDCH') {
        const [session, sub] = sampleId.split('_');
        cueJsonPath = `${datasetConfig.basePath}/${session}/${sub}/cue_detection.json`;
      } else if (dataset === 'CMDC') {
        const match = sampleId.match(/(part\d+)(HC\d+|MDD\d+)(Q\d+)/i);
        if (match) {
          const [, part, subject, q] = match;
          cueJsonPath = `${datasetConfig.basePath}/${part}/${subject}/${q}/cue_detection.json`;
        }
      } else {
        cueJsonPath = `${datasetConfig.basePath}/${sampleId}/cue_detection.json`;
      }

      if (!cueJsonPath) {
        setCurrentSampleIndex(targetIndex);
        setCurrentSample(sampleId);
        setCueData({ cues: [], statistics: {} });
        setOriginalCues([]);
        setAnnotations([]);
        setIsLoading(false);
        return true;
      }

      const response = await fetch(cueJsonPath);
      const data = await response.json();

      setCurrentSampleIndex(targetIndex);
      setCurrentSample(sampleId);
      setCueData(data);
      setOriginalCues(data.cues ? data.cues.map((cue, idx) => ({
        cue_id: cue.id ?? idx,
        text: cue.text,
        original_span: { start: cue.start, end: cue.end }
      })) : []);
      setAnnotations(data.cues ? data.cues.map((cue, idx) => ({
        cue_id: cue.id ?? idx,
        text: cue.text,
        original_span: { start: cue.start, end: cue.end },
        corrected_span: { start: cue.start, end: cue.end }
      })) : []);
      setIsLoading(false);
      return true;
    } catch (err) {
      console.error(`Failed to load ${sampleId}:`, err);
      setCurrentSampleIndex(targetIndex);
      setCurrentSample(sampleId);
      setCueData({ cues: [], statistics: {} });
      setOriginalCues([]);
      setAnnotations([]);
      setIsLoading(false);
      return true;
    }
  }, []);

  const loadNextSample = useCallback((dataset) => {
    const samples = SAMPLE_INDEX[dataset];
    const nextIndex = currentSampleIndex + 1;

    if (nextIndex >= samples.length) {
      alert('This is the last sample.');
      return;
    }

    loadSampleAtIndexStrict(dataset, nextIndex);
  }, [currentSampleIndex, loadSampleAtIndexStrict]);

  const loadPrevSample = useCallback(() => {
    if (!currentDataset || currentSampleIndex <= 0) {
      alert('This is the first sample.');
      return;
    }

    loadSampleAtIndexStrict(currentDataset, currentSampleIndex - 1);
  }, [currentDataset, currentSampleIndex, loadSampleAtIndexStrict]);

  const togglePlay = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const skipBackward = () => {
    if (audioRef.current) {
      audioRef.current.currentTime = Math.max(0, audioRef.current.currentTime - 5);
    }
  };

  const skipForward = () => {
    if (audioRef.current) {
      audioRef.current.currentTime = Math.min(audioDuration, audioRef.current.currentTime + 5);
    }
  };

  const handleSeek = (e) => {
    const time = parseFloat(e.target.value);
    if (audioRef.current) {
      audioRef.current.currentTime = time;
      setAudioTime(time);
    }
  };

  const handleVolumeChange = (e) => {
    const vol = parseFloat(e.target.value);
    setVolume(vol);
    if (audioRef.current) {
      audioRef.current.volume = vol / 20;
    }
  };

  const handlePlaybackRate = (rate) => {
    setPlaybackRate(rate);
    if (audioRef.current) {
      audioRef.current.playbackRate = rate;
    }
  };

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      const currentTime = audioRef.current.currentTime;
      setAudioTime(currentTime);

      // Check if we've reached the cue playback end time
      if (cuePlaybackEnd !== null && currentTime >= cuePlaybackEnd) {
        audioRef.current.pause();
        setIsPlaying(false);
        setCuePlaybackEnd(null);
      }
    }
  };

  const handleLoadedMetadata = () => {
    if (audioRef.current) {
      setAudioDuration(audioRef.current.duration);
      audioRef.current.volume = volume / 20;
      audioRef.current.playbackRate = playbackRate;
    }
  };

  const handleEnded = () => {
    setIsPlaying(false);
  };

  const updateAnnotation = (index, field, value) => {
    const newAnnotations = [...annotations];
    newAnnotations[index] = { ...newAnnotations[index], [field]: value };

    if (field === 'corrected_span') {
      const originalCue = originalCues.find(c => c.cue_id === newAnnotations[index].cue_id);
      if (originalCue) {
        const isModified =
          value.start !== originalCue.original_span.start ||
          value.end !== originalCue.original_span.end;
        newAnnotations[index].status = isModified ? 'modified' : 'unchanged';
      }
    }

    setAnnotations(newAnnotations);
  };

  const deleteAnnotation = (index) => {
    const newAnnotations = [...annotations];
    newAnnotations[index] = { ...newAnnotations[index], status: 'deleted' };
    setAnnotations(newAnnotations);
  };

  const restoreAnnotation = (index) => {
    const newAnnotations = [...annotations];
    const annotation = newAnnotations[index];
    const originalCue = originalCues.find(c => c.cue_id === annotation.cue_id);

    if (originalCue) {
      newAnnotations[index] = {
        ...annotation,
        corrected_span: { ...originalCue.original_span },
        status: 'unchanged'
      };
    }

    setAnnotations(newAnnotations);
  };

  const addNewCue = () => {
    if (!newCueText.trim() || !newCueStart || !newCueEnd) {
      alert('Please complete all cue fields.');
      return;
    }

    const start = parseFloat(newCueStart);
    const end = parseFloat(newCueEnd);

    if (isNaN(start) || isNaN(end) || start >= end) {
      alert('Please enter a valid time range.');
      return;
    }

    const maxId = Math.max(0, ...annotations.map(a => a.cue_id), ...originalCues.map(c => c.cue_id));
    const newCue = {
      cue_id: maxId + 1,
      text: newCueText.trim(),
      original_span: { start, end },
      corrected_span: { start, end },
      status: 'added'
    };

    setAnnotations([...annotations, newCue]);
    setNewCueText('');
    setNewCueStart('');
    setNewCueEnd('');
    setShowAddCueForm(false);
  };

  const completeAnnotation = async () => {
    const activeAnnotations = annotations.filter(a => a.status !== 'deleted');

    const payload = {
      dataset: currentDataset,
      sample_id: currentSample,
      cues: activeAnnotations.map(a => ({
        cue_id: a.cue_id,
        text: a.text,
        original_span: a.original_span,
        corrected_span: a.corrected_span,
        status: a.status || 'unchanged'
      }))
    };

    try {
      const response = await fetch('/api/save-annotation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error('Save failed');
      }

      const result = await response.json();

      const newCompleted = {
        ...completedSamples,
        [currentDataset]: [...(completedSamples[currentDataset] || []), currentSample]
      };
      setCompletedSamples(newCompleted);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(newCompleted));

      setIsAnnotated(true);

      // Automatically advance to the next sample.
      const samples = SAMPLE_INDEX[currentDataset];
      const nextIndex = currentSampleIndex + 1;
      if (nextIndex < samples.length) {
        setTimeout(() => {
          loadSampleAtIndexStrict(currentDataset, nextIndex);
        }, 300);
      } else {
        alert('All samples in this dataset have been completed.');
      }
    } catch (err) {
      console.error('Error saving annotation:', err);
      alert('Save failed. Please try again.');
    }
  };

  const getTimePercentage = (time) => {
    if (!audioDuration) return 0;
    return (time / audioDuration) * 100;
  };

  const activeCueCount = annotations.filter((annotation) => annotation.status !== 'deleted').length;

  if (!currentAnnotator) {
    return (
      <div className="login-container">
        <div className="login-card">
          <h1 className="login-title">Cue Annotation Console</h1>
          <p className="login-subtitle">Select an annotator account to start reviewing timestamps.</p>
          <div className="annotator-list">
            {Object.entries(ANNOTATOR_DATASET).map(([annotator, dataset]) => (
              <button
                key={annotator}
                onClick={() => selectAnnotator(annotator)}
                className="annotator-btn"
              >
                <div className="annotator-info">
                  <div className="annotator-avatar">{annotator}</div>
                  <div className="annotator-details">
                    <div className="annotator-name">Annotator {annotator}</div>
                    <div className="annotator-dataset">Dataset: {dataset}</div>
                  </div>
                </div>
                <span className="annotator-arrow">→</span>
              </button>
            ))}
          </div>
        </div>
      </div>
    );
  }

  const progress = {
    current: currentSampleIndex + 1,
    total: SAMPLE_INDEX[currentDataset]?.length || 0
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <div className="header-left">
            <div className="header-avatar">{currentAnnotator}</div>
            <div className="header-info">
              <h1 className="header-title">Cue Timestamp Review</h1>
              <p className="header-subtitle">
                <span>{currentDataset}</span>
                <span className={`status-badge ${isAnnotated ? 'annotated' : 'unannotated'}`}>
                  {isAnnotated ? 'Saved' : 'Unsaved'}
                </span>
              </p>
            </div>
          </div>
          <div className="header-right">
            <div className="progress-text">
              Progress <span className="progress-current">{progress.current}</span> / {progress.total}
            </div>
            <button onClick={() => { setCurrentAnnotator(null); }} className="logout-btn">
              Sign Out
            </button>
          </div>
        </div>
      </header>

      {/* Audio Player */}
      <div className="audio-player">
        <div className="audio-player-content">
          <div className="sample-nav">
            <button
              onClick={loadPrevSample}
              disabled={currentSampleIndex <= 0}
              className="nav-btn"
            >
              Previous
            </button>
            <span className="sample-id">{currentSample}</span>
            <button
              onClick={() => loadNextSample(currentDataset)}
              className="nav-btn"
            >
              Next
            </button>
          </div>

          <div className="audio-controls">
            <div className="control-buttons">
              <button onClick={skipBackward} className="skip-btn">-5s</button>
              <button onClick={togglePlay} className="play-btn">
                {isPlaying ? '⏸' : '▶'}
              </button>
              <button onClick={skipForward} className="skip-btn">+5s</button>
            </div>

            <div className="progress-section">
              <div className="progress-bar-container">
                <input
                  type="range"
                  min={0}
                  max={audioDuration || 100}
                  value={audioTime}
                  onChange={handleSeek}
                  className="progress-slider"
                />
                {cueData?.cues?.map((cue, idx) => {
                  const start = cue.corrected_span?.start ?? cue.start ?? 0;
                  const end = cue.corrected_span?.end ?? cue.end ?? 0;
                  const left = getTimePercentage(start);
                  const width = getTimePercentage(end) - left;
                  return (
                    <div
                      key={idx}
                      className="cue-marker"
                      style={{
                        left: `${left}%`,
                        width: `${width}%`
                      }}
                      title={`${cue.text} (${start.toFixed(2)}s - ${end.toFixed(2)}s)`}
                    />
                  );
                })}
              </div>
              <div className="time-display">
                <span>{audioTime.toFixed(1)}s</span>
                <span>{audioDuration ? audioDuration.toFixed(1) + 's' : '--'}</span>
              </div>
            </div>

            <div className="current-time-box">
              <div className="time-label">Current Time</div>
              <div className="time-value">{audioTime.toFixed(3)}s</div>
            </div>

            <div className="volume-control">
              <div className="control-header">
                <span>Volume</span>
                <span>{volume.toFixed(1)}x</span>
              </div>
              <input
                type="range"
                min={0}
                max={20}
                step={0.1}
                value={volume}
                onChange={handleVolumeChange}
                className="control-slider"
              />
            </div>

            <div className="speed-control">
              <div className="control-header">
                <span>Speed</span>
                <span>{playbackRate}x</span>
              </div>
              <div className="speed-buttons">
                {[1, 1.2, 1.5, 1.7, 2].map(rate => (
                  <button
                    key={rate}
                    onClick={() => handlePlaybackRate(rate)}
                    className={`speed-btn ${playbackRate === rate ? 'active' : ''}`}
                  >
                    {rate}x
                  </button>
                ))}
              </div>
            </div>

            <button onClick={completeAnnotation} className="complete-btn">
              Save & Next
            </button>
          </div>
        </div>

        {currentSample && (
          <audio
            ref={audioRef}
            key={audioKey}
            src={`${DATASET_CONFIG[currentDataset].audioBasePath}/${DATASET_CONFIG[currentDataset].getAudioPath(currentSample)}`}
            onTimeUpdate={handleTimeUpdate}
            onLoadedMetadata={handleLoadedMetadata}
            onEnded={handleEnded}
            crossOrigin="anonymous"
          />
        )}
      </div>

      {/* Main Content */}
      <main ref={mainRef} className="main-content">
        {isLoading ? (
          <div className="loading-container">
            <div className="loading-spinner"></div>
            <p className="loading-text">Loading sample...</p>
          </div>
        ) : error ? (
          <div className="success-message">
            <div className="success-icon">✓</div>
            <h2>{error}</h2>
            <p>No remaining samples were found for this dataset.</p>
          </div>
        ) : (
          <>
            <div className="stats-bar">
              <div className="stat-item">
                <span className="stat-label">Active Cues</span>
                <span className="stat-value">{activeCueCount}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Word Count</span>
                <span className="stat-value">{cueData?.statistics?.total_words || '-'}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Cue Coverage</span>
                <span className="stat-value">{cueData?.statistics?.cue_time_coverage_sec?.toFixed(2) || '-'}s</span>
              </div>
              <button
                onClick={() => setShowAddCueForm(!showAddCueForm)}
                className="add-cue-toggle"
              >
                {showAddCueForm ? 'Hide Form' : 'Add Cue'}
              </button>
            </div>

            {showAddCueForm && (
              <div className="add-cue-form">
                <h3 className="form-title">Add Cue</h3>
                <div className="form-fields">
                  <div className="form-field">
                    <label>Cue Text</label>
                    <input
                      type="text"
                      value={newCueText}
                      onChange={(e) => setNewCueText(e.target.value)}
                      placeholder="Enter cue text"
                    />
                  </div>
                  <div className="form-field small">
                    <label>Start (s)</label>
                    <input
                      type="number"
                      step="0.001"
                      value={newCueStart}
                      onChange={(e) => {
                        const val = e.target.value;
                        // Limit precision to three decimal places.
                        if (val.includes('.')) {
                          const [intPart, decPart] = val.split('.');
                          if (decPart.length > 3) {
                            setNewCueStart(`${intPart}.${decPart.slice(0, 3)}`);
                            return;
                          }
                        }
                        setNewCueStart(val);
                      }}
                      placeholder="0.000"
                    />
                  </div>
                  <div className="form-field small">
                    <label>End (s)</label>
                    <input
                      type="number"
                      step="0.001"
                      value={newCueEnd}
                      onChange={(e) => {
                        const val = e.target.value;
                        // Limit precision to three decimal places.
                        if (val.includes('.')) {
                          const [intPart, decPart] = val.split('.');
                          if (decPart.length > 3) {
                            setNewCueEnd(`${intPart}.${decPart.slice(0, 3)}`);
                            return;
                          }
                        }
                        setNewCueEnd(val);
                      }}
                      placeholder="0.000"
                    />
                  </div>
                  <div className="form-actions">
                    <button onClick={addNewCue} className="confirm-btn">Add</button>
                    <button onClick={() => setShowAddCueForm(false)} className="cancel-btn">Cancel</button>
                  </div>
                </div>
              </div>
            )}

            <div className="cue-list-container">
              <div className="cue-list-header">
                <h2>Cue Review</h2>
              </div>
              <div className="cue-list">
                {annotations.length === 0 ? (
                  <div className="empty-state">No cue entries available for this sample.</div>
                ) : (
                  annotations.map((annotation, index) => (
                    <div
                      key={annotation.cue_id}
                      className={`cue-item ${annotation.status === 'deleted' ? 'deleted' : ''}`}
                    >
                      {/* Row 1: index, text field, and delete action */}
                      <div className="cue-row cue-row-1">
                        <div className="cue-number">{index + 1}</div>
                        <div className="cue-text-field">
                          <input
                            type="text"
                            value={annotation.text}
                            onChange={(e) => updateAnnotation(index, 'text', e.target.value)}
                            disabled={annotation.status === 'deleted'}
                            placeholder="Cue text"
                          />
                        </div>
                        <div className="cue-actions">
                          {annotation.status === 'deleted' ? (
                            <button
                              onClick={() => restoreAnnotation(index)}
                              className="restore-btn"
                            >
                              Restore
                            </button>
                          ) : (
                            <button
                              onClick={() => deleteAnnotation(index)}
                              className="delete-btn"
                            >
                              Delete
                            </button>
                          )}
                        </div>
                      </div>

                      {/* Row 2: time inputs, playback action, and status */}
                      <div className="cue-row cue-row-2">
                        <div className="cue-time-inputs">
                          <div className="time-input">
                            <label>Start</label>
                            <input
                              type="number"
                              step="0.001"
                              value={annotation.corrected_span?.start ?? ''}
                              onChange={(e) => {
                                const val = parseFloat(e.target.value);
                                updateAnnotation(index, 'corrected_span', {
                                  ...annotation.corrected_span,
                                  start: isNaN(val) ? 0 : val
                                });
                              }}
                              onFocus={(e) => {
                                e.target.select();
                              }}
                              disabled={annotation.status === 'deleted'}
                            />
                          </div>
                          <div className="time-input">
                            <label>End</label>
                            <input
                              type="number"
                              step="0.001"
                              value={annotation.corrected_span?.end ?? ''}
                              onChange={(e) => {
                                const val = parseFloat(e.target.value);
                                updateAnnotation(index, 'corrected_span', {
                                  ...annotation.corrected_span,
                                  end: isNaN(val) ? 0 : val
                                });
                              }}
                              onFocus={(e) => {
                                e.target.select();
                              }}
                              disabled={annotation.status === 'deleted'}
                            />
                          </div>
                        </div>
                        <button
                          onClick={() => {
                            const startTime = annotation.corrected_span?.start ?? annotation.original_span?.start ?? 0;
                            const endTime = annotation.corrected_span?.end ?? annotation.original_span?.end ?? startTime;
                            if (audioRef.current) {
                              audioRef.current.currentTime = startTime;
                              setCuePlaybackEnd(endTime);
                              audioRef.current.play();
                              setIsPlaying(true);
                            }
                          }}
                          className="cue-play-btn"
                          disabled={annotation.status === 'deleted'}
                          title={`Play ${(annotation.corrected_span?.start ?? annotation.original_span?.start ?? 0).toFixed(3)}s - ${(annotation.corrected_span?.end ?? annotation.original_span?.end ?? 0).toFixed(3)}s`}
                        >
                          ▶
                        </button>
                        <div className={`status-tag ${annotation.status || 'unchanged'}`}>
                          {annotation.status === 'deleted' ? 'Deleted' :
                           annotation.status === 'added' ? 'Added' :
                           annotation.status === 'modified' ? 'Modified' : 'Unchanged'}
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}

export default App;
