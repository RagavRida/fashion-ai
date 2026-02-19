import React, { useState, useCallback, useRef } from 'react';
import axios from 'axios';
import './index.css';

const API_BASE = process.env.REACT_APP_API_URL || '';

// ─── Utility: convert File to base64 ─────────────────────────────────────────
const fileToBase64 = (file) =>
    new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => resolve(reader.result.split(',')[1]);
        reader.onerror = reject;
    });

// ─── Suggestions ──────────────────────────────────────────────────────────────
const PROMPT_SUGGESTIONS = [
    'Upcycle a denim jacket into a cropped streetwear jacket with patches',
    'Transform a plain white shirt into a bohemian crop top',
    'Convert an old blazer into a sustainable athleisure jacket',
    'Redesign vintage trousers as high-waisted statement jeans',
];

const REFINE_SUGGESTIONS = [
    'Make sleeves shorter',
    'Change to pastel colors',
    'Add embroidery details',
    'Make waist more fitted',
    'Add distressed texture',
];

const MODES = [
    { id: 'generate', label: '✨ Prompt', desc: 'Generate from text' },
    { id: 'redesign', label: '🖼️ Image', desc: 'Redesign garment' },
    { id: 'redesign_prompt', label: '✏️ Both', desc: 'Image + prompt' },
];

// ─── Components ───────────────────────────────────────────────────────────────
function Spinner() {
    return <div className="spinner" />;
}

function UploadZone({ onFile, preview, onClear }) {
    const [dragging, setDragging] = useState(false);
    const inputRef = useRef();

    const handleDrop = useCallback((e) => {
        e.preventDefault();
        setDragging(false);
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) onFile(file);
    }, [onFile]);

    const handleChange = (e) => {
        const file = e.target.files[0];
        if (file) onFile(file);
    };

    return (
        <div
            className={`upload-zone ${dragging ? 'dragging' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            onClick={() => !preview && inputRef.current?.click()}
        >
            {preview ? (
                <div className="upload-preview">
                    <img src={preview} alt="Uploaded garment" />
                    <button
                        className="upload-clear"
                        onClick={(e) => { e.stopPropagation(); onClear(); }}
                    >✕</button>
                </div>
            ) : (
                <>
                    <span className="upload-icon">👗</span>
                    <p className="upload-text">Drop garment image here</p>
                    <p className="upload-subtext">or click to browse • JPEG, PNG, WebP</p>
                </>
            )}
            <input
                ref={inputRef}
                type="file"
                accept="image/*"
                onChange={handleChange}
                style={{ display: 'none' }}
            />
        </div>
    );
}

function Gallery({ images, loading, onRefine, onSetBase, generationTime }) {
    if (loading) {
        return (
            <div className="gallery-grid">
                {[...Array(4)].map((_, i) => (
                    <div key={i} className="skeleton" />
                ))}
            </div>
        );
    }

    if (images.length === 0) {
        return (
            <div className="gallery-grid">
                <div className="gallery-empty">
                    <div className="gallery-empty-icon">🎨</div>
                    <p className="gallery-empty-text">
                        Your AI-generated fashion designs will appear here.<br />
                        Choose a mode and click Generate to begin.
                    </p>
                </div>
            </div>
        );
    }

    return (
        <>
            {generationTime && (
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 12, textAlign: 'right' }}>
                    Generated in {generationTime}s
                </div>
            )}
            <div className="gallery-grid">
                {images.map((img, i) => (
                    <div key={i} className="gallery-item">
                        <img
                            className="gallery-img"
                            src={`data:image/jpeg;base64,${img.image_b64}`}
                            alt={`Generated design ${i + 1}`}
                        />
                        <div className="gallery-overlay">
                            <button className="gallery-action-btn" onClick={() => onRefine(img)}>
                                ✏️ Refine
                            </button>
                            <button className="gallery-action-btn primary" onClick={() => onSetBase(img)}>
                                📥 Use
                            </button>
                        </div>
                    </div>
                ))}
            </div>
        </>
    );
}

function RefinementPanel({ selectedImage, onNewImages }) {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [originalPrompt] = useState('');

    const sendRefinement = async (refinementText) => {
        if (!selectedImage || !refinementText.trim() || loading) return;

        const userMsg = { role: 'user', text: refinementText };
        setMessages((m) => [...m, userMsg]);
        setInput('');
        setLoading(true);

        try {
            const res = await axios.post(`${API_BASE}/refine`, {
                previous_image_b64: selectedImage.image_b64,
                refinement_prompt: refinementText,
                original_prompt: originalPrompt,
                n_images: 4,
            });
            onNewImages(res.data.images, res.data.generation_time_secs);
            setMessages((m) => [
                ...m,
                { role: 'assistant', text: `Applied: "${refinementText}". ${res.data.images.length} new designs generated.` },
            ]);
        } catch (err) {
            setMessages((m) => [
                ...m,
                { role: 'assistant', text: `Error: ${err.response?.data?.detail || err.message}` },
            ]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="panel mt-4">
            <div className="panel-title">
                <span className="panel-title-icon">✏️</span>
                Refinement Chat
            </div>

            {!selectedImage ? (
                <p style={{ fontSize: 13, color: 'var(--text-muted)', textAlign: 'center', padding: '16px 0' }}>
                    Click "Refine" on any generated image to refine it
                </p>
            ) : (
                <>
                    <div style={{ marginBottom: 12, padding: '8px 12px', background: 'rgba(167,139,250,0.07)', borderRadius: 10, border: '1px solid rgba(167,139,250,0.2)', display: 'flex', alignItems: 'center', gap: 10 }}>
                        <img
                            src={`data:image/jpeg;base64,${selectedImage.image_b64}`}
                            alt="Selected"
                            style={{ width: 40, height: 40, borderRadius: 6, objectFit: 'cover' }}
                        />
                        <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                            Refining selected design
                        </span>
                    </div>

                    <div className="refine-suggestions">
                        {REFINE_SUGGESTIONS.map((s) => (
                            <button key={s} className="prompt-suggestion" onClick={() => sendRefinement(s)}>
                                {s}
                            </button>
                        ))}
                    </div>

                    {messages.length > 0 && (
                        <div className="chat-history">
                            {messages.map((msg, i) => (
                                <div key={i} className={`chat-msg ${msg.role}`}>
                                    {msg.text}
                                </div>
                            ))}
                        </div>
                    )}

                    <div className="refine-input-row">
                        <input
                            className="refine-input"
                            placeholder="e.g. Make sleeves shorter, add floral embroidery…"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && sendRefinement(input)}
                            disabled={loading}
                        />
                        <button
                            className="refine-send"
                            onClick={() => sendRefinement(input)}
                            disabled={loading || !input.trim()}
                        >
                            {loading ? <Spinner /> : '→'}
                        </button>
                    </div>
                </>
            )}
        </div>
    );
}

function DIYPanel({ lastImages, lastPrompt }) {
    const [guide, setGuide] = useState(null);
    const [loading, setLoading] = useState(false);
    const [expanded, setExpanded] = useState(false);
    const [garmentInput, setGarmentInput] = useState('');

    const fetchGuide = async () => {
        setLoading(true);
        setExpanded(true);
        try {
            const res = await axios.post(`${API_BASE}/diy_guide`, {
                garment_category: garmentInput || 'garment',
                edits_applied: lastPrompt
                    ? lastPrompt.split(',').map((s) => s.trim()).filter(Boolean)
                    : [],
                style_description: lastPrompt || 'modern upcycled fashion',
                difficulty_target: 'Medium',
                final_image_b64: lastImages[0]?.image_b64 || null,
            });
            setGuide(res.data);
        } catch (err) {
            alert(`DIY guide error: ${err.response?.data?.detail || err.message}`);
        } finally {
            setLoading(false);
        }
    };

    const diffClass = (diff) => {
        if (!diff) return '';
        const d = diff.toLowerCase();
        if (d === 'easy') return 'diff-easy';
        if (d === 'hard') return 'diff-hard';
        return 'diff-medium';
    };

    return (
        <div className="diy-panel">
            <div className="diy-header" onClick={() => setExpanded(!expanded)}>
                <div className="diy-header-left">
                    <div className="diy-header-icon">🛠️</div>
                    <div>
                        <div className="diy-header-title">DIY Household Guide</div>
                        <div className="diy-header-sub">Step-by-step upcycling instructions</div>
                    </div>
                </div>
                <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                    <input
                        style={{
                            background: 'var(--bg-input)',
                            border: '1px solid var(--border-subtle)',
                            borderRadius: 10,
                            padding: '8px 14px',
                            color: 'var(--text-primary)',
                            fontSize: 13,
                            fontFamily: 'inherit',
                            outline: 'none',
                            width: 160,
                        }}
                        placeholder="Garment type…"
                        value={garmentInput}
                        onChange={(e) => setGarmentInput(e.target.value)}
                        onClick={(e) => e.stopPropagation()}
                    />
                    <button
                        className="diy-get-btn"
                        onClick={(e) => { e.stopPropagation(); fetchGuide(); }}
                        disabled={loading}
                    >
                        {loading ? <Spinner /> : '✨ Get Guide'}
                    </button>
                </div>
            </div>

            {expanded && guide && (
                <div className="diy-body">
                    <div className="diy-title-section">
                        <h2>{guide.title}</h2>
                        <p>{guide.edits_summary}</p>
                    </div>

                    <div className="diy-meta">
                        <span className="diy-meta-badge time">⏱ {guide.estimated_time}</span>
                        <span className={`diy-meta-badge ${diffClass(guide.difficulty)}`}>
                            ⚡ {guide.difficulty}
                        </span>
                    </div>

                    <div>
                        <div className="diy-section-title">Materials</div>
                        <div className="diy-list">
                            {guide.materials.map((m, i) => (
                                <span key={i} className="diy-tag">🧵 {m}</span>
                            ))}
                        </div>
                    </div>

                    <div>
                        <div className="diy-section-title">Tools</div>
                        <div className="diy-list">
                            {guide.tools.map((t, i) => (
                                <span key={i} className="diy-tag">🔧 {t}</span>
                            ))}
                        </div>
                    </div>

                    <div>
                        <div className="diy-section-title">Steps</div>
                        <div className="diy-steps">
                            {guide.steps.map((s, i) => (
                                <div key={i} className="diy-step">
                                    <div className="diy-step-num">{s.step}</div>
                                    <div className="diy-step-body">
                                        <p className="diy-step-instruction">{s.instruction}</p>
                                        {s.tip && (
                                            <p className="diy-step-tip">💡 <em>{s.tip}</em></p>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div>
                        <div className="diy-section-title">Safety & Budget Tips</div>
                        <div className="diy-tip-grid">
                            {guide.safety_tips.map((t, i) => (
                                <div key={`s${i}`} className="diy-tip-item">🛡️ {t}</div>
                            ))}
                            {guide.budget_tips.map((t, i) => (
                                <div key={`b${i}`} className="diy-tip-item">💰 {t}</div>
                            ))}
                        </div>
                    </div>

                    <div>
                        <div className="diy-section-title">Sustainability Benefits</div>
                        <div className="diy-list">
                            {guide.sustainability_benefits.map((b, i) => (
                                <span key={i} className="diy-tag">🌿 {b}</span>
                            ))}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
    const [mode, setMode] = useState('generate');
    const [prompt, setPrompt] = useState('');
    const [uploadedFile, setUploadedFile] = useState(null);
    const [uploadedPreview, setUploadedPreview] = useState(null);
    const [images, setImages] = useState([]);
    const [loading, setLoading] = useState(false);
    const [selectedForRefine, setSelectedForRefine] = useState(null);
    const [generationTime, setGenerationTime] = useState(null);
    const [lastPrompt, setLastPrompt] = useState('');
    const [error, setError] = useState('');

    const handleFileUpload = useCallback((file) => {
        setUploadedFile(file);
        const url = URL.createObjectURL(file);
        setUploadedPreview(url);
    }, []);

    const handleClearUpload = () => {
        setUploadedFile(null);
        setUploadedPreview(null);
    };

    const generate = async () => {
        setError('');
        setLoading(true);
        setImages([]);
        setGenerationTime(null);

        try {
            let res;
            if (mode === 'generate') {
                if (!prompt.trim()) { setError('Please enter a prompt.'); setLoading(false); return; }
                res = await axios.post(`${API_BASE}/generate`, { prompt, n_images: 4 });
            } else if (mode === 'redesign') {
                if (!uploadedFile) { setError('Please upload a garment image.'); setLoading(false); return; }
                const b64 = await fileToBase64(uploadedFile);
                res = await axios.post(`${API_BASE}/redesign`, { image_b64: b64, n_images: 4 });
            } else {
                if (!uploadedFile || !prompt.trim()) { setError('Please upload an image and enter a prompt.'); setLoading(false); return; }
                const b64 = await fileToBase64(uploadedFile);
                res = await axios.post(`${API_BASE}/redesign_prompt`, { image_b64: b64, prompt, n_images: 4 });
            }

            setImages(res.data.images || []);
            setGenerationTime(res.data.generation_time_secs);
            setLastPrompt(prompt);
        } catch (err) {
            setError(err.response?.data?.detail || err.message || 'Generation failed');
        } finally {
            setLoading(false);
        }
    };

    const handleRefineResult = (newImages, time) => {
        setImages(newImages);
        setGenerationTime(time);
    };

    return (
        <div className="app">
            {/* Header */}
            <header className="header">
                <div className="header-logo">
                    <div className="header-logo-icon">♻</div>
                    <div>
                        <div className="header-title">Fashion Reuse Studio</div>
                        <div className="header-subtitle">AI-Powered Upcycling System</div>
                    </div>
                </div>
                <div className="header-badge">Beta v1.0</div>
            </header>

            {/* Main Grid */}
            <div className="main-grid">
                {/* Left Panel — Controls */}
                <div>
                    <div className="panel">
                        <div className="panel-title">
                            <span className="panel-title-icon">✨</span>
                            Design Mode
                        </div>

                        {/* Mode Tabs */}
                        <div className="mode-tabs">
                            {MODES.map((m) => (
                                <button
                                    key={m.id}
                                    className={`mode-tab ${mode === m.id ? 'active' : ''}`}
                                    onClick={() => setMode(m.id)}
                                >
                                    {m.label}
                                </button>
                            ))}
                        </div>

                        {/* Upload (show for image modes) */}
                        {(mode === 'redesign' || mode === 'redesign_prompt') && (
                            <>
                                <div className="prompt-label mb-2">Garment Image</div>
                                <UploadZone
                                    onFile={handleFileUpload}
                                    preview={uploadedPreview}
                                    onClear={handleClearUpload}
                                />
                                <div className="divider" />
                            </>
                        )}

                        {/* Prompt (show for prompt modes) */}
                        {(mode === 'generate' || mode === 'redesign_prompt') && (
                            <>
                                <div className="prompt-label mb-2">Design Prompt</div>
                                <textarea
                                    className="prompt-textarea"
                                    placeholder="Describe the garment style, upcycling transformation, materials…"
                                    value={prompt}
                                    onChange={(e) => setPrompt(e.target.value)}
                                    rows={4}
                                />
                                <div className="prompt-suggestions">
                                    {PROMPT_SUGGESTIONS.slice(0, 3).map((s) => (
                                        <button
                                            key={s}
                                            className="prompt-suggestion"
                                            onClick={() => setPrompt(s)}
                                        >
                                            {s.substring(0, 35)}…
                                        </button>
                                    ))}
                                </div>
                            </>
                        )}

                        {/* Error */}
                        {error && (
                            <div style={{
                                marginTop: 12, padding: '10px 14px',
                                background: 'rgba(248,113,113,0.1)',
                                border: '1px solid rgba(248,113,113,0.3)',
                                borderRadius: 10, fontSize: 13,
                                color: '#f87171',
                            }}>
                                ⚠️ {error}
                            </div>
                        )}

                        {/* Generate Button */}
                        <button
                            className={`generate-btn ${loading ? 'loading' : ''}`}
                            onClick={generate}
                            disabled={loading}
                        >
                            {loading ? (
                                <><Spinner /> Generating…</>
                            ) : (
                                <>✨ Generate Designs</>
                            )}
                        </button>
                    </div>

                    {/* Refinement Chat */}
                    <RefinementPanel
                        selectedImage={selectedForRefine}
                        onNewImages={handleRefineResult}
                    />
                </div>

                {/* Right Panel — Gallery */}
                <div>
                    <div className="panel">
                        <div className="gallery-header">
                            <span className="gallery-title">
                                <span style={{ marginRight: 6, fontSize: 16 }}>🖼️</span>
                                Generated Designs
                            </span>
                            {images.length > 0 && (
                                <span className="gallery-count">{images.length} results</span>
                            )}
                        </div>

                        <Gallery
                            images={images}
                            loading={loading}
                            generationTime={generationTime}
                            onRefine={(img) => setSelectedForRefine(img)}
                            onSetBase={(img) => {
                                // Convert b64 back to file-like for redesign
                                const blob = b64ToBlob(img.image_b64);
                                const file = new File([blob], 'selected_design.jpg', { type: 'image/jpeg' });
                                handleFileUpload(file);
                                setMode('redesign_prompt');
                            }}
                        />
                    </div>

                    {/* DIY Guide */}
                    <DIYPanel lastImages={images} lastPrompt={lastPrompt} />
                </div>
            </div>
        </div>
    );
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
function b64ToBlob(b64Data, contentType = 'image/jpeg', sliceSize = 512) {
    const byteCharacters = atob(b64Data);
    const byteArrays = [];
    for (let offset = 0; offset < byteCharacters.length; offset += sliceSize) {
        const slice = byteCharacters.slice(offset, offset + sliceSize);
        const byteNumbers = new Array(slice.length).fill(0).map((_, i) => slice.charCodeAt(i));
        byteArrays.push(new Uint8Array(byteNumbers));
    }
    return new Blob(byteArrays, { type: contentType });
}
