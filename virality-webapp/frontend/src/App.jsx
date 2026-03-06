import { useEffect, useMemo, useRef, useState } from 'react';

const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');

const CORE_FIELD_ORDER = [
  'title',
  'description',
  'duration_seconds',
  'channel_country',
  'default_language',
  'published_at',
  'published_dayofweek',
  'published_hour',
  'channel_subscriber_count'
];

const FIELD_LABEL_OVERRIDES = {
  title: 'Title',
  description: 'Description',
  duration_seconds: 'Duration (sec)',
  channel_country: 'Country',
  default_language: 'Language',
  default_audio_language: 'Language',
  published_at: 'Publish date',
  published_dayofweek: 'Publish day',
  published_hour: 'Published hour',
  channel_subscriber_count: 'Channel subscribers',
  query: 'Query'
};

const FIELD_PLACEHOLDER_OVERRIDES = {
  title: 'Paste the Shorts title',
  description: 'Optional description text',
  duration_seconds: 'e.g. 27',
  published_at: '',
  published_dayofweek: '',
  published_hour: '',
  channel_subscriber_count: 'e.g. 125000',
  query: 'Optional seed query'
};

const DAY_OF_WEEK_OPTIONS = [
  { label: 'Monday', value: '0' },
  { label: 'Tuesday', value: '1' },
  { label: 'Wednesday', value: '2' },
  { label: 'Thursday', value: '3' },
  { label: 'Friday', value: '4' },
  { label: 'Saturday', value: '5' },
  { label: 'Sunday', value: '6' }
];

function buildApiPath(path) {
  if (!API_BASE) return path;
  return `${API_BASE}${path}`;
}

function formatRaw(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return Math.round(Number(value)).toLocaleString();
}

function parseNumber(value) {
  if (value === null || value === undefined || value === '') return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function strategyLabel(mode) {
  return mode === 'fast' ? 'Concat' : 'Per-model best';
}

function normalizeMetadataForSubmit(metadata) {
  const out = {};
  for (const [key, value] of Object.entries(metadata)) {
    if (typeof value === 'string') {
      out[key] = value.trim();
    } else {
      out[key] = value;
    }
  }
  return out;
}

function classifyBooleanField(name) {
  const k = String(name).toLowerCase();
  if (k.includes('channel') || k.includes('subscriber') || k.includes('video_count') || k.includes('view_count')) {
    return 'channel';
  }
  if (k.includes('thumb') || k.includes('aspect') || k.includes('width') || k.includes('height') || k.includes('text_present')) {
    return 'thumbnail';
  }
  return 'content';
}

function prettyFieldLabel(name) {
  if (FIELD_LABEL_OVERRIDES[name]) return FIELD_LABEL_OVERRIDES[name];
  return name
    .replaceAll('_', ' ')
    .replace(/\b[a-z]/g, (m) => m.toUpperCase());
}

function fieldPlaceholder(field) {
  if (FIELD_PLACEHOLDER_OVERRIDES[field.name]) return FIELD_PLACEHOLDER_OVERRIDES[field.name];
  if (field.type === 'number') return 'Optional';
  return '';
}

function toYyyyMmDd(date) {
  const year = date.getFullYear();
  const month = `${date.getMonth() + 1}`.padStart(2, '0');
  const day = `${date.getDate()}`.padStart(2, '0');
  return `${year}-${month}-${day}`;
}

function dateInputValueFromStored(field, value) {
  if (value === null || value === undefined || value === '') return '';
  if (field.type === 'number') {
    const n = Number(value);
    if (!Number.isFinite(n)) return '';
    const ms = Math.abs(n) > 1e11 ? n : n * 1000;
    const date = new Date(ms);
    if (Number.isNaN(date.getTime())) return '';
    return toYyyyMmDd(date);
  }

  const text = String(value).trim();
  if (!text) return '';
  if (/^\d{4}-\d{2}-\d{2}$/.test(text)) return text;
  const parsed = new Date(text);
  if (Number.isNaN(parsed.getTime())) return '';
  return toYyyyMmDd(parsed);
}

function hourToTimeValue(value) {
  if (value === null || value === undefined || value === '') return '';
  const parsedNumber = Number(value);
  if (Number.isFinite(parsedNumber)) {
    const hour = Math.max(0, Math.min(23, Math.trunc(parsedNumber)));
    return `${String(hour).padStart(2, '0')}:00`;
  }
  const text = String(value).trim();
  if (/^\d{2}:\d{2}$/.test(text)) return text;
  return '';
}

function isProvidedFieldValue(field, value) {
  if (value === null || value === undefined) return false;

  if (field.type === 'number') {
    const n = Number(value);
    return Number.isFinite(n);
  }

  if (field.type === 'boolean') {
    const text = String(value).trim().toLowerCase();
    return value === true || value === 1 || text === 'true' || text === '1';
  }

  return String(value).trim().length > 0;
}

function metadataConfidenceLabel(fields, metadata) {
  if (!metadata || !fields || fields.length === 0) return 'low';

  // Exclude booleans from denominator because many default to false.
  const scored = fields.filter((field) => field.type !== 'boolean');
  if (scored.length === 0) return 'low';

  let provided = 0;
  for (const field of scored) {
    if (isProvidedFieldValue(field, metadata[field.name])) provided += 1;
  }

  const coverage = provided / scored.length;
  if (coverage >= 0.85) return 'high';
  if (coverage >= 0.6) return 'medium-high';
  if (coverage >= 0.35) return 'medium';
  if (coverage >= 0.15) return 'medium-low';
  return 'low';
}

function confidenceToProgress(confidence) {
  const table = {
    low: 24,
    'medium-low': 42,
    medium: 58,
    'medium-high': 74,
    high: 90
  };
  return table[String(confidence || '').toLowerCase()] || 0;
}

function FieldInput({ field, value, onChange, compact = false, className = '' }) {
  const id = `field-${field.name}`;
  const label = prettyFieldLabel(field.name);
  const shellClass = `vp-field ${compact ? 'compact' : ''} ${field.name === 'description' ? 'wide' : ''} ${className}`.trim();

  if (field.type === 'boolean') {
    return (
      <label htmlFor={id} className="vp-toggle-row">
        <span>{label}</span>
        <input
          id={id}
          type="checkbox"
          checked={Boolean(value)}
          onChange={(e) => onChange(field.name, e.target.checked)}
        />
      </label>
    );
  }

  if (field.options && field.options.length > 0) {
    const normalizedOptions = field.options
      .map((opt) => String(opt).trim())
      .filter((opt) => opt.length > 0 && opt.toLowerCase() !== 'auto-detect');

    return (
      <label htmlFor={id} className={shellClass}>
        <span>{label}</span>
        <select id={id} value={value ?? ''} onChange={(e) => onChange(field.name, e.target.value)}>
          <option value="">Auto-detect</option>
          {normalizedOptions.map((opt) => (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ))}
        </select>
      </label>
    );
  }

  if (field.name === 'published_dayofweek') {
    return (
      <label htmlFor={id} className={shellClass}>
        <span>{label}</span>
        <select id={id} value={value ?? ''} onChange={(e) => onChange(field.name, e.target.value)}>
          <option value="">Auto-detect</option>
          {DAY_OF_WEEK_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      </label>
    );
  }

  if (field.name === 'published_at') {
    const inputValue = dateInputValueFromStored(field, value);
    return (
      <label htmlFor={id} className={shellClass}>
        <span>{label}</span>
        <input
          id={id}
          type="date"
          value={inputValue}
          onChange={(e) => {
            const dateText = e.target.value;
            if (!dateText) {
              onChange(field.name, '');
              return;
            }
            if (field.type === 'number') {
              const seconds = Math.floor(new Date(`${dateText}T00:00:00`).getTime() / 1000);
              onChange(field.name, String(seconds));
              return;
            }
            onChange(field.name, dateText);
          }}
        />
      </label>
    );
  }

  if (field.name === 'published_hour') {
    const inputValue = hourToTimeValue(value);
    return (
      <label htmlFor={id} className={shellClass}>
        <span>{label}</span>
        <input
          id={id}
          type="time"
          step={60}
          value={inputValue}
          onChange={(e) => {
            const timeText = e.target.value;
            if (!timeText) {
              onChange(field.name, '');
              return;
            }
            const hour = Number(timeText.split(':')[0]);
            if (!Number.isFinite(hour)) {
              onChange(field.name, '');
              return;
            }
            if (field.type === 'number') {
              onChange(field.name, String(hour));
            } else {
              onChange(field.name, timeText);
            }
          }}
        />
      </label>
    );
  }

  const isTextArea = field.name === 'description';
  const inputType = field.type === 'number' ? 'number' : 'text';

  return (
    <label htmlFor={id} className={shellClass}>
      <span>{label}</span>
      {isTextArea ? (
        <textarea
          id={id}
          rows={4}
          value={value ?? ''}
          placeholder={fieldPlaceholder(field)}
          onChange={(e) => onChange(field.name, e.target.value)}
        />
      ) : (
        <input
          id={id}
          type={inputType}
          step={field.type === 'number' ? 'any' : undefined}
          value={value ?? ''}
          placeholder={fieldPlaceholder(field)}
          onChange={(e) => onChange(field.name, e.target.value)}
        />
      )}
    </label>
  );
}

function ForecastCard({ title, value, confidence, progress, rangeLabel }) {
  const hasValue = value !== null && value !== undefined && !Number.isNaN(Number(value));
  const fillWidth = hasValue ? Math.max(8, Math.min(100, progress || 0)) : 0;
  return (
    <div className="vp-forecast-card">
      <div className="vp-forecast-title">{title}</div>
      <div className="vp-forecast-value">{hasValue ? formatRaw(value) : ''}</div>
      <div className="vp-forecast-sub">Expected views</div>
      <div className="vp-progress-track">
        <div className="vp-progress-fill" style={{ width: `${fillWidth}%` }} />
      </div>
      {confidence ? <div className="vp-forecast-confidence">Confidence: {confidence}</div> : null}
      {rangeLabel ? <div className="vp-forecast-range">Range: {rangeLabel}</div> : null}
    </div>
  );
}

export default function App() {
  const [schema, setSchema] = useState(null);
  const [mode, setMode] = useState('fast');
  const [modeInfoOpen, setModeInfoOpen] = useState(false);
  const [metadata, setMetadata] = useState({});
  const [videoFile, setVideoFile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [lastSubmittedMetadata, setLastSubmittedMetadata] = useState(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    let active = true;

    async function loadSchema() {
      try {
        const res = await fetch(buildApiPath('/api/schema'));
        if (!res.ok) {
          let detail = '';
          try {
            const payload = await res.json();
            detail = payload?.detail ? String(payload.detail) : '';
          } catch (_err) {
            detail = '';
          }
          throw new Error(detail ? `Schema request failed (${res.status}): ${detail}` : `Schema request failed (${res.status})`);
        }

        const data = await res.json();
        if (!active) return;
        setSchema(data);

        const defaults = {};
        for (const field of data.fields || []) {
          if (field.default !== undefined && field.default !== null) {
            defaults[field.name] = field.default;
          } else if (field.type === 'boolean') {
            defaults[field.name] = false;
          } else {
            defaults[field.name] = '';
          }
        }
        setMetadata(defaults);
      } catch (err) {
        if (!active) return;
        setError(String(err));
      } finally {
        if (active) setLoading(false);
      }
    }

    loadSchema();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!modeInfoOpen) return undefined;
    function onKeydown(event) {
      if (event.key === 'Escape') {
        setModeInfoOpen(false);
      }
    }
    window.addEventListener('keydown', onKeydown);
    return () => window.removeEventListener('keydown', onKeydown);
  }, [modeInfoOpen]);

  const allFields = useMemo(() => schema?.fields || [], [schema]);
  const fieldMap = useMemo(() => {
    const map = new Map();
    for (const field of allFields) map.set(field.name, field);
    return map;
  }, [allFields]);

  const coreFields = useMemo(() => {
    const ordered = CORE_FIELD_ORDER.filter((name) => fieldMap.has(name)).map((name) => fieldMap.get(name));
    return ordered;
  }, [fieldMap]);

  const coreFieldMap = useMemo(() => {
    const map = new Map();
    for (const field of coreFields) map.set(field.name, field);
    return map;
  }, [coreFields]);

  const primaryCoreKeys = useMemo(
    () =>
      new Set([
        'title',
        'description',
        'duration_seconds',
        'channel_country',
        'default_language',
        'published_at',
        'published_dayofweek',
        'published_hour',
        'channel_subscriber_count'
      ]),
    []
  );

  const primaryCoreFields = useMemo(() => {
    const out = {};
    for (const key of primaryCoreKeys) {
      const field = coreFieldMap.get(key);
      if (field) out[key] = field;
    }
    return out;
  }, [coreFieldMap, primaryCoreKeys]);

  const overflowCoreFields = useMemo(
    () => coreFields.filter((field) => !primaryCoreKeys.has(field.name)),
    [coreFields, primaryCoreKeys]
  );

  const advancedFields = useMemo(() => {
    const coreNames = new Set(coreFields.map((f) => f.name));
    return allFields.filter((f) => !coreNames.has(f.name));
  }, [allFields, coreFields]);

  const advancedBooleanGroups = useMemo(() => {
    const groups = { content: [], thumbnail: [], channel: [] };
    for (const field of advancedFields) {
      if (field.type !== 'boolean') continue;
      groups[classifyBooleanField(field.name)].push(field);
    }
    return groups;
  }, [advancedFields]);

  const advancedOtherFields = useMemo(() => advancedFields.filter((f) => f.type !== 'boolean'), [advancedFields]);

  const advancedGroupCards = useMemo(
    () => [
      { key: 'content', title: 'Content signals', fields: advancedBooleanGroups.content },
      { key: 'thumbnail', title: 'Thumbnail signals', fields: advancedBooleanGroups.thumbnail },
      { key: 'channel', title: 'Channel signals', fields: advancedBooleanGroups.channel }
    ],
    [advancedBooleanGroups]
  );

  function updateField(name, value) {
    setMetadata((prev) => ({ ...prev, [name]: value }));
  }

  async function autofillFromFile() {
    if (!videoFile) {
      setError('Choose an MP4 first, then click Autofill from file.');
      return;
    }

    const next = { ...metadata };
    if (!next.title) {
      next.title = videoFile.name.replace(/\.[^.]+$/, '');
    }

    if ('duration_seconds' in next) {
      try {
        const objectUrl = URL.createObjectURL(videoFile);
        const duration = await new Promise((resolve, reject) => {
          const video = document.createElement('video');
          video.preload = 'metadata';
          video.onloadedmetadata = () => {
            resolve(video.duration);
            URL.revokeObjectURL(objectUrl);
          };
          video.onerror = () => {
            reject(new Error('Could not read video metadata'));
            URL.revokeObjectURL(objectUrl);
          };
          video.src = objectUrl;
        });
        if (Number.isFinite(duration)) {
          next.duration_seconds = String(Math.max(0, Math.round(Number(duration))));
        }
      } catch (_err) {
        // Keep manual entry path if browser metadata extraction fails.
      }
    }

    setMetadata(next);
  }

  async function onSubmit(event) {
    event.preventDefault();
    setError('');
    setResult(null);

    if (!videoFile) {
      setError('Please choose an MP4 file.');
      return;
    }

    try {
      setSubmitting(true);
      const normalizedMetadata = normalizeMetadataForSubmit(metadata);
      const body = new FormData();
      body.append('video_file', videoFile);
      body.append('mode', mode);
      body.append('metadata_json', JSON.stringify(normalizedMetadata));

      const res = await fetch(buildApiPath('/api/predict'), {
        method: 'POST',
        body
      });

      let payload = null;
      try {
        payload = await res.json();
      } catch (_err) {
        payload = null;
      }

      if (!res.ok) {
        const detail = payload?.detail ? String(payload.detail) : '';
        throw new Error(detail ? `Prediction failed (${res.status}): ${detail}` : `Prediction failed (${res.status})`);
      }

      setResult(payload);
      setLastSubmittedMetadata(normalizedMetadata);
    } catch (err) {
      const text = String(err || '');
      if (text.includes('ETIMEDOUT') || text.includes('Failed to fetch')) {
        setError(`${text}\nBackend may not be ready yet. Wait for backend startup logs, then retry.`);
      } else {
        setError(text);
      }
    } finally {
      setSubmitting(false);
    }
  }

  const metadataConfidence = useMemo(
    () => metadataConfidenceLabel(allFields, lastSubmittedMetadata || {}),
    [allFields, lastSubmittedMetadata]
  );

  const prediction7 = useMemo(() => {
    if (!result) return null;
    const progressFromConfidence = confidenceToProgress(metadataConfidence);
    if (result.mode === 'fast') {
      return {
        raw: parseNumber(result.predictions_7d?.prediction_raw),
        minRaw: null,
        maxRaw: null,
        confidence: metadataConfidence,
        progress: progressFromConfidence
      };
    }

    const minRaw = parseNumber(result.range_7d?.min_raw);
    const maxRaw = parseNumber(result.range_7d?.max_raw);
    const value = minRaw != null && maxRaw != null ? (minRaw + maxRaw) / 2 : null;
    return {
      raw: value,
      minRaw,
      maxRaw,
      confidence: metadataConfidence,
      progress: progressFromConfidence
    };
  }, [result, metadataConfidence]);

  const prediction30 = useMemo(() => {
    if (!result) return null;
    const progressFromConfidence = confidenceToProgress(metadataConfidence);
    if (result.mode === 'fast') {
      return {
        raw: parseNumber(result.predictions_30d?.prediction_raw),
        minRaw: null,
        maxRaw: null,
        confidence: metadataConfidence,
        progress: progressFromConfidence
      };
    }

    const minRaw = parseNumber(result.range_30d?.min_raw);
    const maxRaw = parseNumber(result.range_30d?.max_raw);
    const value = minRaw != null && maxRaw != null ? (minRaw + maxRaw) / 2 : null;
    return {
      raw: value,
      minRaw,
      maxRaw,
      confidence: metadataConfidence,
      progress: progressFromConfidence
    };
  }, [result, metadataConfidence]);

  const impactNotes = useMemo(() => {
    const notes = [];
    const isVertical = Boolean(metadata?.is_vertical_thumb);
    const duration = parseNumber(metadata?.duration_seconds);
    const missingGeo = !metadata?.channel_country || !metadata?.default_language;

    if (isVertical) notes.push('Strong vertical thumbnail signal');
    if (duration != null && duration <= 35) notes.push('Short duration helps retention prior');
    if (missingGeo) notes.push('Missing language/country lowered certainty');
    if (result?.transcript?.text_present === 0) notes.push('Transcript unavailable, text confidence reduced');

    if (notes.length === 0) {
      notes.push('Add core metadata to improve confidence');
      notes.push('Use Autofill from file for quick defaults');
    }
    return notes.slice(0, 3);
  }, [metadata, result]);

  const fullModeRows = useMemo(() => {
    if (!result || result.mode !== 'full') return [];

    const rows7 = Array.isArray(result.predictions_7d) ? result.predictions_7d : [];
    const rows30 = Array.isArray(result.predictions_30d) ? result.predictions_30d : [];
    const keyed = new Map();

    for (const row of rows7) {
      const key = `${row.model}::${row.strategy}::${row.run_id}`;
      keyed.set(key, {
        model: String(row.model || ''),
        strategy: String(row.strategy || ''),
        runId: String(row.run_id || ''),
        p7Raw: parseNumber(row.prediction_raw),
        p30Raw: null
      });
    }
    for (const row of rows30) {
      const key = `${row.model}::${row.strategy}::${row.run_id}`;
      const existing = keyed.get(key) || {
        model: String(row.model || ''),
        strategy: String(row.strategy || ''),
        runId: String(row.run_id || ''),
        p7Raw: null,
        p30Raw: null
      };
      existing.p30Raw = parseNumber(row.prediction_raw);
      keyed.set(key, existing);
    }

    return Array.from(keyed.values());
  }, [result]);

  return (
    <div className="vp-page">
      <div className="vp-shell">
        <header className="vp-hero">
          <div className="vp-hero-left">
            <div className="vp-pill">MP4 to Virality Predictor</div>
            <h1>Predict 7-day and 30-day Shorts performance</h1>
            <p>Upload a video, fill only the important inputs, and let the system auto-detect the rest.</p>
          </div>
          <div className="vp-mode-box">
            <label>
              <div className="vp-mode-head">
                <span>Mode</span>
                <button
                  type="button"
                  className="vp-info-btn"
                  aria-label="Explain Fast and Full mode"
                  onClick={() => setModeInfoOpen(true)}
                >
                  i
                </button>
              </div>
              <select value={mode} onChange={(e) => setMode(e.target.value)}>
                {(schema?.modes || ['fast', 'full']).map((m) => (
                  <option key={m} value={m}>
                    {m === 'fast' ? 'Fast' : 'Full'}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span>Fusion strategy</span>
              <div className="vp-mode-static">{strategyLabel(mode)}</div>
            </label>
          </div>
        </header>

        {loading ? <div className="vp-loading">Loading schema...</div> : null}
        {error ? <div className="vp-error">{error}</div> : null}

        {!loading && schema ? (
          <form className="vp-layout" onSubmit={onSubmit}>
            <section className="vp-core-card">
              <div className="vp-section-head">
                <div>
                  <h2>Core inputs</h2>
                  <p>Show the highest-value fields first. Everything else goes under Advanced.</p>
                </div>
                <button type="button" className="vp-secondary-btn" onClick={autofillFromFile}>
                  Autofill from file
                </button>
              </div>

              <div className="vp-core-grid">
                {primaryCoreFields.title ? (
                  <FieldInput
                    key={primaryCoreFields.title.name}
                    field={primaryCoreFields.title}
                    value={metadata[primaryCoreFields.title.name]}
                    onChange={updateField}
                    className="wide"
                  />
                ) : null}

                {primaryCoreFields.description ? (
                  <FieldInput
                    key={primaryCoreFields.description.name}
                    field={primaryCoreFields.description}
                    value={metadata[primaryCoreFields.description.name]}
                    onChange={updateField}
                    className="wide"
                  />
                ) : null}

                <div className="vp-file-field">
                  <label>MP4 Upload</label>
                  <button
                    type="button"
                    className="vp-dropzone"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <strong>{videoFile ? videoFile.name : 'Drop MP4 here or browse'}</strong>
                    <span>Auto-extract duration, thumbnail size, and basic metadata</span>
                  </button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="video/mp4"
                    className="vp-file-hidden"
                    onChange={(e) => setVideoFile(e.target.files?.[0] || null)}
                  />
                </div>

                {primaryCoreFields.duration_seconds ? (
                  <FieldInput
                    key={primaryCoreFields.duration_seconds.name}
                    field={primaryCoreFields.duration_seconds}
                    value={metadata[primaryCoreFields.duration_seconds.name]}
                    onChange={updateField}
                  />
                ) : null}

                {primaryCoreFields.channel_country ? (
                  <FieldInput
                    key={primaryCoreFields.channel_country.name}
                    field={primaryCoreFields.channel_country}
                    value={metadata[primaryCoreFields.channel_country.name]}
                    onChange={updateField}
                  />
                ) : null}

                {primaryCoreFields.default_language ? (
                  <FieldInput
                    key={primaryCoreFields.default_language.name}
                    field={primaryCoreFields.default_language}
                    value={metadata[primaryCoreFields.default_language.name]}
                    onChange={updateField}
                  />
                ) : null}

                {primaryCoreFields.published_at ? (
                  <FieldInput
                    key={primaryCoreFields.published_at.name}
                    field={primaryCoreFields.published_at}
                    value={metadata[primaryCoreFields.published_at.name]}
                    onChange={updateField}
                  />
                ) : null}

                {primaryCoreFields.channel_subscriber_count ? (
                  <FieldInput
                    key={primaryCoreFields.channel_subscriber_count.name}
                    field={primaryCoreFields.channel_subscriber_count}
                    value={metadata[primaryCoreFields.channel_subscriber_count.name]}
                    onChange={updateField}
                  />
                ) : null}

                {primaryCoreFields.published_dayofweek ? (
                  <FieldInput
                    key={primaryCoreFields.published_dayofweek.name}
                    field={primaryCoreFields.published_dayofweek}
                    value={metadata[primaryCoreFields.published_dayofweek.name]}
                    onChange={updateField}
                  />
                ) : null}

                {primaryCoreFields.published_hour ? (
                  <FieldInput
                    key={primaryCoreFields.published_hour.name}
                    field={primaryCoreFields.published_hour}
                    value={metadata[primaryCoreFields.published_hour.name]}
                    onChange={updateField}
                  />
                ) : null}

                {overflowCoreFields.map((field) => (
                  <FieldInput key={field.name} field={field} value={metadata[field.name]} onChange={updateField} />
                ))}
              </div>

              <div className="vp-warning-box">
                Missing values can default to model priors. Mark fields as optional unless they materially change prediction quality.
              </div>

              <div className="vp-advanced-wrap">
                <div className="vp-section-head">
                  <div>
                    <h3>Advanced features</h3>
                    <p>Only for manual overrides and debugging</p>
                  </div>
                  <button type="button" className="vp-link-btn" onClick={() => setAdvancedOpen((v) => !v)}>
                    {advancedOpen ? 'Hide' : 'Show'}
                  </button>
                </div>

                {advancedOpen ? (
                  <>
                    <div className="vp-advanced-groups">
                      {advancedGroupCards
                        .filter((group) => group.fields.length > 0)
                        .map((group) => (
                          <div key={group.key} className="vp-adv-card">
                            <h4>{group.title}</h4>
                            <div className="vp-adv-list">
                              {group.fields.map((field) => (
                                <FieldInput
                                  key={field.name}
                                  field={field}
                                  value={metadata[field.name]}
                                  onChange={updateField}
                                  compact
                                />
                              ))}
                            </div>
                          </div>
                        ))}
                    </div>

                    {advancedOtherFields.length > 0 ? (
                      <div className="vp-advanced-other-grid">
                        {advancedOtherFields.map((field) => (
                          <FieldInput
                            key={field.name}
                            field={field}
                            value={metadata[field.name]}
                            onChange={updateField}
                            compact
                          />
                        ))}
                      </div>
                    ) : null}
                  </>
                ) : null}
              </div>
            </section>

            <aside className="vp-summary-card">
              <h2>Prediction summary</h2>
              <p>Run prediction once your core fields are complete.</p>

              <button type="submit" className="vp-primary-btn" disabled={submitting || loading}>
                {submitting ? 'Running...' : 'Run prediction'}
              </button>

              <div className="vp-summary-grid">
                <ForecastCard
                  title="7-day forecast"
                  value={prediction7?.raw}
                  confidence={prediction7?.confidence}
                  progress={prediction7?.progress || 0}
                  rangeLabel={
                    prediction7?.minRaw != null && prediction7?.maxRaw != null
                      ? `${formatRaw(prediction7.minRaw)} - ${formatRaw(prediction7.maxRaw)}`
                      : ''
                  }
                />
                <ForecastCard
                  title="30-day forecast"
                  value={prediction30?.raw}
                  confidence={prediction30?.confidence}
                  progress={prediction30?.progress || 0}
                  rangeLabel={
                    prediction30?.minRaw != null && prediction30?.maxRaw != null
                      ? `${formatRaw(prediction30.minRaw)} - ${formatRaw(prediction30.maxRaw)}`
                      : ''
                  }
                />
              </div>

              <div className="vp-impact-box">
                <h4>What changed the score most?</h4>
                <ul>
                  {impactNotes.map((note) => (
                    <li key={note}>• {note}</li>
                  ))}
                </ul>
              </div>

              {fullModeRows.length > 0 ? (
                <div className="vp-model-table-box">
                  <h4>Full mode model outputs</h4>
                  <div className="vp-model-table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>Model</th>
                          <th>Strategy</th>
                          <th>7d</th>
                          <th>30d</th>
                        </tr>
                      </thead>
                      <tbody>
                        {fullModeRows.map((row) => (
                          <tr key={`${row.model}-${row.strategy}-${row.runId}`}>
                            <td>{row.model}</td>
                            <td>{row.strategy}</td>
                            <td>{formatRaw(row.p7Raw)}</td>
                            <td>{formatRaw(row.p30Raw)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ) : null}
            </aside>
          </form>
        ) : null}

        {modeInfoOpen ? (
          <div className="vp-modal-backdrop" role="presentation" onClick={() => setModeInfoOpen(false)}>
            <section
              className="vp-modal"
              role="dialog"
              aria-modal="true"
              aria-labelledby="vp-mode-info-title"
              onClick={(event) => event.stopPropagation()}
            >
              <div className="vp-modal-head">
                <h3 id="vp-mode-info-title">Fast vs Full mode</h3>
                <button type="button" className="vp-modal-close" onClick={() => setModeInfoOpen(false)}>
                  Close
                </button>
              </div>
              <div className="vp-modal-body">
                <h4>Fast mode</h4>
                <p>
                  Uses one lightweight model setup and returns a single prediction for 7-day and 30-day views.
                  Choose this when you want a quicker result.
                </p>

                <h4>Full mode</h4>
                <p>
                  Runs multiple model types and shows each model's prediction, plus a safer overall range.
                  Choose this when you want a more cautious estimate.
                </p>

                <h4>How the full-mode range is computed</h4>
                <ul>
                  <li>We sort model predictions and reduce the impact of extreme outliers.</li>
                  <li>We focus on the middle predictions, then add extra padding on both sides.</li>
                  <li>This creates a more stable range that is less likely to overreact to one model.</li>
                  <li>The final range is converted back to normal view counts and never goes below 0.</li>
                </ul>
              </div>
            </section>
          </div>
        ) : null}
      </div>
    </div>
  );
}
