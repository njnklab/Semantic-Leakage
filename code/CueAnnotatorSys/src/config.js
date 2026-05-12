// Dataset configuration
export const DATASET_CONFIG = {
  'E-DAIC': {
    annotator: 'A1',
    basePath: '/agent/E-DAIC',
    audioBasePath: '/data/E-DAIC',
    getAudioPath: (sampleId) => {
      const base = sampleId.replace('_AUDIO', '');
      return `${base}_P/${sampleId}.wav`;
    }
  },
  'ManDIC': {
    annotator: 'A2',
    basePath: '/agent/ManDIC',
    audioBasePath: '/data/ManDIC/data',
    getAudioPath: (sampleId) => `${sampleId}.WAV`
  },
  'PDCH': {
    annotator: 'A3',
    basePath: '/agent/PDCH',
    audioBasePath: '/data/PDCH',
    getAudioPath: (sampleId) => {
      const [session, sub] = sampleId.split('_');
      return `${session}/${sub}.wav`;
    }
  },
  'CMDC': {
    annotator: 'A4',
    basePath: '/agent/CMDC',
    audioBasePath: '/data/CMDC_EULA',
    getAudioPath: (sampleId) => {
      // Sample ID format: part1HC01Q1
      const match = sampleId.match(/(part\d+)(HC\d+|MDD\d+)(Q\d+)/i);
      if (match) {
        const [, part, subject, q] = match;
        return `${part}/${subject}/${q}.wav`;
      }
      return `${sampleId}.wav`;
    }
  }
};

// Annotator-to-dataset mapping
export const ANNOTATOR_DATASET = {
  'A1': 'E-DAIC',
  'A2': 'ManDIC',
  'A3': 'PDCH',
  'A4': 'CMDC'
};

// Local storage key
export const STORAGE_KEY = 'cue_annotator_completed';
