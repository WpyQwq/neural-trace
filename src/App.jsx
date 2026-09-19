import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity,
  ArrowDown,
  BrainCircuit,
  ChevronDown,
  CircleHelp,
  CirclePause,
  CirclePlay,
  Cpu,
  Database,
  GitBranch,
  Layers3,
  MoreHorizontal,
  Paperclip,
  Pause,
  Play,
  Radio,
  RefreshCw,
  Search,
  Send,
  Settings2,
  SlidersHorizontal,
  Sparkles,
  TerminalSquare,
  TimerReset,
  Waypoints,
  X,
} from 'lucide-react'

const Trace3DCanvas = lazy(() => import('./Trace3DCanvas.jsx'))

const TOKENS = ['请', '解', '释', '量', '子', '纠', '缠']

const STAGES = [
  { id: 'stage-01', index: '01', title: 'L01—04', type: 'hybrid', caption: '3L + 1A' },
  { id: 'stage-02', index: '02', title: 'L05—08', type: 'hybrid', caption: '3L + 1A' },
  { id: 'stage-03', index: '03', title: 'L09—12', type: 'hybrid', caption: '3L + 1A' },
  { id: 'stage-04', index: '04', title: 'L13—16', type: 'hybrid', caption: '3L + 1A' },
  { id: 'stage-05', index: '05', title: 'L17—20', type: 'hybrid', caption: '3L + 1A' },
  { id: 'stage-06', index: '06', title: 'L21—24', type: 'hybrid', caption: '3L + 1A' },
  { id: 'stage-07', index: '07', title: 'L25—28', type: 'hybrid', caption: '3L + 1A' },
  { id: 'stage-08', index: '08', title: 'L29—32', type: 'hybrid', caption: '3L + 1A' },
]

const EVENT_SEED = [
  { time: '00:12.84', text: 'residual stream merged', tone: 'cyan' },
  { time: '00:12.77', text: 'attention pattern stabilized', tone: 'amber' },
  { time: '00:12.68', text: 'token 03 → layer 08', tone: 'cyan' },
  { time: '00:12.51', text: 'delta state updated', tone: 'muted' },
  { time: '00:12.36', text: 'input embedding ready', tone: 'muted' },
]

const DEFAULT_ARCHITECTURE = {
  num_layers: 32,
  pattern: '8 × (3L + 1A)',
  hidden_size: 2560,
  vocab_size: 248320,
  context_length: 262144,
  intermediate_size: 9216,
  attention_heads: 16,
  attention_kv_heads: 4,
  attention_head_dim: 256,
  vision_encoder: true,
  sparse_moe: true,
  mtp: true,
  precision: 'BF16 / INT4',
}

const compactNumber = (value) => {
  const number = Number(value)
  if (!Number.isFinite(number)) return '—'
  if (number >= 1000000) return `${(number / 1000000).toFixed(number % 1000000 ? 1 : 0)}M`
  if (number >= 100000) return `${Math.round(number / 1000)}K`
  return number.toLocaleString('en-US')
}

function IconButton({ label, children, active = false, onClick }) {
  return (
    <button className={`icon-button ${active ? 'is-active' : ''}`} aria-label={label} title={label} onClick={onClick}>
      {children}
    </button>
  )
}

function StatusDot({ tone = 'green' }) {
  return <span className={`status-dot status-${tone}`} aria-hidden="true" />
}

function App() {
  const [isPlaying, setIsPlaying] = useState(true)
  const [step, setStep] = useState(128)
  const [activeToken, setActiveToken] = useState(3)
  const [selectedLayer, setSelectedLayer] = useState('L08')
  const [speed, setSpeed] = useState(1)
  const [prompt, setPrompt] = useState('解释一下量子纠缠')
  const [attachment, setAttachment] = useState(null)
  const [responseText, setResponseText] = useState('')
  const [responseSource, setResponseSource] = useState('demo')
  const [attentionValues, setAttentionValues] = useState(null)
  const [liveSignal, setLiveSignal] = useState({ kind: 'waiting', value: null, mean: null, shape: null })
  const [events, setEvents] = useState(EVENT_SEED)
  const [runtime, setRuntime] = useState({ connected: false, model: 'Qwen3.5-4B', modelPath: 'waiting for local model', architecture: DEFAULT_ARCHITECTURE })
  const [activeView, setActiveView] = useState('trace')
  const [spaceMode, setSpaceMode] = useState('3d')
  const [processMode, setProcessMode] = useState('forward')
  const [showHistory, setShowHistory] = useState(false)
  const [notice, setNotice] = useState('')
  const noticeTimer = useRef(null)
  const streamAbortRef = useRef(null)
  const traceRef = useRef({ step, activeToken, selectedLayer })
  traceRef.current = { step, activeToken, selectedLayer }

  useEffect(() => {
    let mounted = true
    const refreshRuntime = () => {
      fetch('/api/health')
        .then((response) => response.json())
        .then((data) => {
          if (mounted) setRuntime(data)
        })
        .catch(() => {})
    }
    refreshRuntime()
    const timer = window.setInterval(refreshRuntime, 3500)
    return () => {
      mounted = false
      window.clearInterval(timer)
    }
  }, [])

  useEffect(() => {
    if (!isPlaying) return undefined
    const timer = window.setInterval(() => {
      setStep((current) => (current >= 512 ? 0 : current + 1))
      setActiveToken((current) => (current + 1) % TOKENS.length)
    }, Math.max(180, 850 / speed))
    return () => window.clearInterval(timer)
  }, [isPlaying, speed])

  useEffect(() => {
    const timer = window.setInterval(() => {
      const currentTrace = traceRef.current
      setEvents((current) => {
        const next = [
          { time: `00:${String(12 + (currentTrace.step % 40)).padStart(2, '0')}.${String(currentTrace.step % 100).padStart(2, '0')}`, text: `${TOKENS[currentTrace.activeToken]} → ${currentTrace.selectedLayer}`, tone: currentTrace.activeToken % 2 ? 'amber' : 'cyan' },
          ...current,
        ]
        return next.slice(0, 5)
      })
    }, 1800)
    return () => window.clearInterval(timer)
  }, [])

  useEffect(() => () => streamAbortRef.current?.abort(), [])

  const notify = (message) => {
    setNotice(message)
    window.clearTimeout(noticeTimer.current)
    noticeTimer.current = window.setTimeout(() => setNotice(''), 2600)
  }

  const handleAttachment = (event) => {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    if (file.size > 4 * 1024 * 1024) {
      notify('图片需小于 4 MB')
      return
    }
    const reader = new FileReader()
    reader.onload = () => setAttachment({ name: file.name, dataUrl: String(reader.result) })
    reader.readAsDataURL(file)
  }

  const consumeTraceStream = async (tracePrompt, traceMode, traceAttachment = null) => {
    streamAbortRef.current?.abort()
    const controller = new AbortController()
    streamAbortRef.current = controller
    try {
      const requestUrl = traceAttachment ? '/api/stream' : `/api/stream?prompt=${encodeURIComponent(tracePrompt)}&mode=${traceMode}`
      const requestOptions = traceAttachment ? {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: tracePrompt, mode: traceMode, image_data: traceAttachment.dataUrl }),
      } : {}
      const response = await fetch(requestUrl, { ...requestOptions, signal: controller.signal })
      if (!response.ok || !response.body) return
      setIsPlaying(false)
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      while (true) {
        const { value, done } = await reader.read()
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done })
        const blocks = buffer.split('\n\n')
        buffer = blocks.pop() || ''
        for (const block of blocks) {
          const line = block.split('\n').find((entry) => entry.startsWith('data: '))
          if (!line) continue
          const payload = JSON.parse(line.slice(6))
          const tokenIndex = TOKENS.indexOf(payload.token)
          setStep(Math.min(512, payload.step * 11))
          setActiveToken(tokenIndex >= 0 ? tokenIndex : payload.step % TOKENS.length)
          if (payload.layer) setSelectedLayer(payload.layer)
          if (payload.text) {
            setResponseText(payload.text)
            setResponseSource(payload.source === 'model_runtime' ? 'model' : 'demo')
          }
          if (payload.attention) setAttentionValues(payload.attention)
          if (payload.kind && payload.kind !== 'runtime_error' && payload.kind !== 'generation') {
            setLiveSignal({
              kind: payload.kind,
              value: Number.isFinite(Number(payload.value)) ? Number(payload.value) : null,
              mean: Number.isFinite(Number(payload.mean)) ? Number(payload.mean) : null,
              shape: Array.isArray(payload.shape) ? payload.shape : null,
            })
          }
          if (payload.kind === 'runtime_error') notify(`真实模型运行错误：${payload.error}`)
          const eventText = payload.kind === 'runtime_error'
            ? `runtime error · ${payload.error}`
            : [payload.kind, payload.layer].filter(Boolean).join(' · ')
          const eventTone = ['full_attention', 'attention_weight', 'attention_output', 'gradient', 'loss', 'logits', 'runtime_error', 'attention_unavailable', 'vision_input', 'vision_encoder', 'router_weights', 'expert_mixture', 'mtp_logits'].includes(payload.kind) ? 'amber' : 'cyan'
          setEvents((current) => [{ time: 'stream', text: eventText, tone: eventTone }, ...current].slice(0, 5))
        }
        if (done) break
      }
      notify('遥测流已完成')
    } catch (error) {
      if (error.name !== 'AbortError') notify('后端未连接，继续使用本地模拟')
    }
  }

  const submitPrompt = (event) => {
    event.preventDefault()
    if (!prompt.trim()) return
    setStep(0)
    setActiveToken(0)
    setIsPlaying(true)
    setResponseText('')
    setResponseSource('demo')
    setAttentionValues(null)
    setLiveSignal({ kind: 'waiting', value: null, mean: null, shape: null })
    setEvents((current) => [{ time: '00:00.00', text: `trace started · ${prompt.trim()}`, tone: 'cyan' }, ...current].slice(0, 5))
    notify(processMode === 'backward' ? '反向传播观测已开始' : '新一轮计算观测已开始')
    void consumeTraceStream(prompt.trim(), processMode, attachment)
  }

  const progress = (step / 512) * 100
  const graphState = useMemo(() => ({
    activeStage: Math.floor(step / 64) % STAGES.length,
    pulse: (step % 32) / 32,
  }), [step])
  const selectedLayerNumber = Number(selectedLayer.match(/\d+/)?.[0] || 8)
  const selectedNodeDescription = selectedLayer === 'OUT' ? 'logits projection' : selectedLayer === 'AUX' ? 'router / expert telemetry' : selectedLayer.includes('—') ? 'hybrid block · 3L + 1A' : selectedLayerNumber % 4 === 0 ? 'full attention block' : 'DeltaNet block'
  const processLabel = { forward: 'FORWARD PASS', backward: 'BACKWARD PASS', attention: 'ATTENTION WEIGHTS', state: 'DELTA STATE' }[processMode]
  const architecture = { ...DEFAULT_ARCHITECTURE, ...(runtime.architecture || {}) }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <div className="brand-mark"><BrainCircuit size={16} strokeWidth={1.8} /></div>
          <span className="brand-name">NEURAL TRACE</span>
          <span className="brand-divider" />
          <span className="brand-context">LOCAL OBSERVABILITY</span>
        </div>
        <div className="topbar-actions">
          <div className="runtime-status">
            <StatusDot tone={runtime.connected ? 'green' : runtime.downloaded ? 'cyan' : 'amber'} />
            <span>{runtime.connected ? 'MODEL READY' : runtime.downloaded ? 'CHECKPOINT READY' : 'DEMO TELEMETRY'}</span>
          </div>
          <div className="model-select"><Cpu size={14} /><span>{runtime.model}</span><ChevronDown size={13} /></div>
          <IconButton label="Help"><CircleHelp size={16} /></IconButton>
          <IconButton label="Settings"><Settings2 size={16} /></IconButton>
        </div>
      </header>

      <div className="workspace">
        <aside className={`sidebar ${showHistory ? 'is-open' : ''}`}>
          <div className="sidebar-head">
            <span>SESSIONS</span>
            <IconButton label="More session options"><MoreHorizontal size={16} /></IconButton>
          </div>
          <form className="new-trace" onSubmit={submitPrompt}>
            <label htmlFor="prompt">Start a trace</label>
            <div className="prompt-field">
              <input id="prompt" value={prompt} onChange={(event) => setPrompt(event.target.value)} />
              <label className="attachment-button" title="Attach image" aria-label="Attach image"><Paperclip size={14} /><input type="file" accept="image/*" onChange={handleAttachment} /></label>
              <button type="submit" aria-label="Start trace"><Send size={14} /></button>
            </div>
            {attachment && <div className="attachment-chip"><span><Paperclip size={12} />{attachment.name}</span><button type="button" onClick={() => setAttachment(null)} aria-label="Remove image"><X size={12} /></button></div>}
          </form>
          <div className="session-list">
            <button className="session-row is-selected" onClick={() => notify('当前 trace 已选中')}>
              <div className="session-icon"><Activity size={15} /></div>
              <div className="session-copy"><strong>量子纠缠 · trace 04</strong><span>现在 · 128 / 512 steps</span></div>
              <span className="session-live"><StatusDot tone="green" /></span>
            </button>
            <button className="session-row" onClick={() => notify('历史 trace 还原功能即将接入')}>
              <div className="session-icon muted"><GitBranch size={15} /></div>
              <div className="session-copy"><strong>DeltaNet 状态比较</strong><span>昨天 · 384 steps</span></div>
            </button>
            <button className="session-row" onClick={() => notify('历史 trace 还原功能即将接入')}>
              <div className="session-icon muted"><Waypoints size={15} /></div>
              <div className="session-copy"><strong>长上下文流向</strong><span>8 月 30 日 · 1.2k steps</span></div>
            </button>
          </div>
          <div className="sidebar-footer">
            <div className="storage-line"><Database size={14} /><span>LOCAL CACHE</span><span className="storage-value">4.06 GB</span></div>
            <div className="storage-bar"><span style={{ width: '36%' }} /></div>
            <div className="path-note">{runtime.modelPath || 'D:\\watch\\_LLM\\_think'}</div>
          </div>
        </aside>

        <main className="main-stage">
          <div className="stage-header">
            <div>
              <div className="crumb"><span>TRACE</span><ArrowDown size={12} /><span>{processLabel}</span></div>
              <h1>Token flow</h1>
            </div>
            <div className="stage-header-actions">
              <div className="view-switcher" role="tablist" aria-label="Trace view">
                <button className={activeView === 'trace' && spaceMode === '2d' ? 'is-selected' : ''} onClick={() => { setActiveView('trace'); setSpaceMode('2d') }}>2D Graph</button>
                <button className={activeView === 'trace' && spaceMode === '3d' ? 'is-selected' : ''} onClick={() => { setActiveView('trace'); setSpaceMode('3d') }}>3D Space</button>
                <button className={activeView === 'weights' ? 'is-selected' : ''} onClick={() => setActiveView('weights')}>Weights</button>
              </div>
              <IconButton label="Reset trace" onClick={() => { setStep(0); setActiveToken(0); notify('trace 已重置') }}><RefreshCw size={15} /></IconButton>
            </div>
          </div>

          <div className="token-strip">
            <span className="strip-label">INPUT TOKENS</span>
            <div className="token-list">
              {TOKENS.map((token, index) => <button key={token + index} className={`token ${activeToken === index ? 'is-active' : ''}`} onClick={() => setActiveToken(index)}>{token}</button>)}
            </div>
            <span className="token-count">7 TOKENS</span>
          </div>

          <div className="process-strip">
            <span className="strip-label">PROCESS</span>
            <div className="process-switcher" role="tablist" aria-label="Observable compute process">
              <button className={processMode === 'forward' ? 'is-selected' : ''} onClick={() => setProcessMode('forward')}>Forward</button>
              <button className={processMode === 'backward' ? 'is-selected' : ''} onClick={() => setProcessMode('backward')}>Backward</button>
              <button className={processMode === 'attention' ? 'is-selected' : ''} onClick={() => setProcessMode('attention')}>Attention</button>
              <button className={processMode === 'state' ? 'is-selected' : ''} onClick={() => setProcessMode('state')}>State</button>
            </div>
            <span className="process-note">{processMode === 'backward' ? 'gradient / credit assignment' : processMode === 'attention' ? 'token-to-token routing' : processMode === 'state' ? 'recurrent state update' : 'activation / logits'}</span>
          </div>

          {activeView === 'trace' ? (
            spaceMode === '3d' ? (
              <Suspense fallback={<div className="three-d-panel is-loading"><RefreshCw size={17} className="spin" /><span>Loading 3D trace space</span></div>}>
                <Trace3DCanvas layerCount={Number(architecture.num_layers) || 32} pattern={architecture.pattern} hasImage={Boolean(attachment)} signalKind={liveSignal.kind} step={step} selectedLayer={selectedLayer} setSelectedLayer={setSelectedLayer} activeToken={activeToken} mode={processMode} signalValue={liveSignal.value} />
              </Suspense>
            ) : (
              <TraceGraph graphState={graphState} selectedLayer={selectedLayer} setSelectedLayer={setSelectedLayer} activeToken={activeToken} />
            )
          ) : (
            <WeightsView onSelect={() => notify('权重检查需要完整精度模型')} />
          )}

          <div className="bottom-grid">
            <div className="activation-panel panel-line">
              <div className="panel-heading"><span>{processMode === 'backward' ? 'GRADIENT FIELD' : 'ACTIVATION FIELD'}</span><span className="panel-meta">{selectedLayer} · {processMode === 'backward' ? 'credit signal' : 'hidden state'}</span></div>
              <NeuronField step={step} />
            </div>
            <div className="output-panel panel-line">
              <div className="panel-heading"><span>OUTPUT DISTRIBUTION</span><span className="panel-meta">next token</span></div>
              <div className="output-row"><span className="output-token">纠</span><div className="output-bar"><span style={{ width: '68%' }} /></div><span className="output-value">0.68</span></div>
              <div className="output-row"><span className="output-token">缠</span><div className="output-bar"><span style={{ width: '22%' }} /></div><span className="output-value">0.22</span></div>
              <div className="output-row"><span className="output-token">是</span><div className="output-bar"><span style={{ width: '07%' }} /></div><span className="output-value">0.07</span></div>
            </div>
          </div>

          <TraceTransport isPlaying={isPlaying} setIsPlaying={setIsPlaying} speed={speed} setSpeed={setSpeed} progress={progress} step={step} setStep={setStep} processLabel={processLabel} />
        </main>

        <aside className="inspector">
          <div className="inspector-tabs">
            <button className="is-selected">INSPECTOR</button>
            <button>EVENTS <span className="tab-count">{events.length}</span></button>
          </div>
          <div className="inspector-scroll">
            <section className="inspect-section selected-node">
              <div className="section-kicker">SELECTED NODE</div>
              <div className="node-title-row"><div className="node-symbol hybrid"><Layers3 size={17} /></div><div><h2>{selectedLayer}</h2><p>{selectedNodeDescription}</p></div><span className="node-live">ACTIVE</span></div>
              <div className="metric-grid"><Metric label="HEADS" value={architecture.attention_heads} /><Metric label="KV HEADS" value={architecture.attention_kv_heads} /><Metric label="HEAD DIM" value={architecture.attention_head_dim} /><Metric label="RESIDUAL" value={compactNumber(architecture.hidden_size)} /></div>
              <div className="signal-readout"><div><span>LAST EVENT</span><strong>{liveSignal.kind}</strong></div><div><span>RMS</span><strong>{liveSignal.value == null ? '—' : liveSignal.value.toFixed(4)}</strong></div><div><span>MEAN</span><strong>{liveSignal.mean == null ? '—' : liveSignal.mean.toFixed(4)}</strong></div><div><span>SHAPE</span><strong>{liveSignal.shape?.length ? liveSignal.shape.join(' × ') : '—'}</strong></div></div>
            </section>

            <section className="inspect-section">
              <div className="section-heading"><span>ATTENTION MAP</span><span className="section-hint">token × token</span></div>
              <AttentionMap activeToken={activeToken} values={attentionValues} />
              <div className="legend"><span><i className="legend-swatch cyan" />active path</span><span><i className="legend-swatch gray" />context</span></div>
            </section>

            <section className="inspect-section">
              <div className="section-heading"><span>LIVE EVENTS</span><span className="section-hint">streaming</span></div>
              <div className="event-list">
                {events.map((event, index) => <div className="event-row" key={`${event.time}-${index}`}><span className={`event-dot ${event.tone}`} /><span className="event-time">{event.time}</span><span className="event-text">{event.text}</span></div>)}
              </div>
            </section>

            <section className="inspect-section response-section">
              <div className="section-heading"><span>ASSISTANT OUTPUT</span><span className="section-hint">{responseSource}</span></div>
              <div className={`response-copy ${responseText ? 'has-output' : ''}`}>{responseText || '等待生成事件…'}</div>
              {!responseText && <div className="response-caret" />}
            </section>

            <section className="inspect-section architecture-note">
              <div className="section-heading"><span>MODEL ARCHITECTURE</span><IconButton label="Architecture info"><CircleHelp size={14} /></IconButton></div>
              <div className="architecture-line"><span>Layers</span><strong>{architecture.num_layers}</strong></div>
              <div className="architecture-line"><span>Pattern</span><strong>{architecture.pattern}</strong></div>
              <div className="architecture-line"><span>Context</span><strong>{compactNumber(architecture.context_length)}</strong></div>
              <div className="architecture-line"><span>FFN</span><strong>{compactNumber(architecture.intermediate_size)}</strong></div>
              <div className="architecture-line"><span>Vocabulary</span><strong>{compactNumber(architecture.vocab_size)}</strong></div>
              <div className="architecture-line"><span>Vision encoder</span><strong>{architecture.vision_encoder ? 'unified' : '—'}</strong></div>
              <div className="architecture-line"><span>Sparse MoE</span><strong>{architecture.sparse_moe ? 'router + experts' : '—'}</strong></div>
              <div className="architecture-line"><span>MTP head</span><strong>{architecture.mtp ? 'trained' : '—'}</strong></div>
              <div className="architecture-line"><span>Precision</span><strong>{architecture.precision}</strong></div>
            </section>
          </div>
        </aside>
      </div>

      <button className="mobile-history" onClick={() => setShowHistory((current) => !current)} aria-label="Toggle sessions"><Search size={17} /></button>
      {notice && <div className="toast"><Radio size={14} />{notice}<button onClick={() => setNotice('')} aria-label="Dismiss"><X size={13} /></button></div>}
    </div>
  )
}

function TraceGraph({ graphState, selectedLayer, setSelectedLayer, activeToken }) {
  const packetOffset = graphState.pulse * 100
  const selectedLayerNumber = Number(selectedLayer.match(/\d+/)?.[0] || 0)
  return (
    <section className="graph-panel">
      <div className="graph-toolbar"><div className="graph-legend"><span><i className="legend-swatch cyan" />data flow</span><span><i className="legend-swatch amber" />attention</span><span><i className="legend-swatch gray" />residual</span></div><div className="graph-coordinates">X 04.82&nbsp;&nbsp; Y 08.16&nbsp;&nbsp; Z 00.00</div></div>
      <div className="graph-canvas">
        <div className="grid-overlay" />
        <svg className="flow-svg" viewBox="0 0 1080 390" preserveAspectRatio="none" aria-label="Model architecture data flow">
          <defs>
            <linearGradient id="cyan-flow" x1="0" x2="1"><stop offset="0" stopColor="#78e4e8" stopOpacity="0.18" /><stop offset="1" stopColor="#78e4e8" stopOpacity="0.8" /></linearGradient>
            <linearGradient id="amber-flow" x1="0" x2="1"><stop offset="0" stopColor="#efb56b" stopOpacity="0.12" /><stop offset="1" stopColor="#efb56b" stopOpacity="0.75" /></linearGradient>
          </defs>
          <path className="flow-line residual" d="M72 194 H1008" />
          {STAGES.map((stage, index) => {
            const x = 150 + index * 112
            const active = graphState.activeStage === index
            return <g key={stage.id}>
              <path className="flow-line data" d={`M${x - 44} 194 H${x + 44}`} />
              {active && <circle className={`flow-packet ${index % 2 ? 'amber' : ''}`} cx={x - 44 + packetOffset * 0.88} cy="194" r="4" />}
              <line className="branch-line" x1={x} y1="194" x2={x} y2="124" />
              <line className="branch-line" x1={x} y1="194" x2={x} y2="264" />
              <circle className={`stage-port ${active ? 'active' : ''}`} cx={x} cy="194" r="7" />
              <circle className="branch-port" cx={x} cy="124" r="3" />
              <circle className="branch-port" cx={x} cy="264" r="3" />
            </g>
          })}
          <path className="flow-line data" d="M32 194 H108" />
          <path className="flow-line data" d="M972 194 H1048" />
          <circle className="stage-port input" cx="32" cy="194" r="7" />
          <circle className="stage-port output" cx="1048" cy="194" r="7" />
        </svg>
        <div className="flow-node input-node"><span className="node-index">IN</span><strong>embeddings</strong><small>7 × 2,560</small></div>
        {STAGES.map((stage, index) => {
          const active = graphState.activeStage === index
          const selected = selectedLayer === stage.title || (selectedLayerNumber >= index * 4 + 1 && selectedLayerNumber <= index * 4 + 4)
          return <button key={stage.id} className={`flow-node stage-node ${stage.type} ${active ? 'active' : ''} ${selected ? 'selected' : ''}`} style={{ left: `${11.1 + index * 10.37}%` }} onClick={() => setSelectedLayer(stage.title)}>
            <span className="node-index">{stage.index}</span><strong>{stage.title}</strong><span className="micro-layers" aria-label="3 DeltaNet layers and 1 full attention layer"><i /><i /><i /><i className="attention-mini" /></span><small>{stage.caption}</small>
          </button>
        })}
        <div className="flow-node output-node"><span className="node-index">OUT</span><strong>logits</strong><small>248,320 dim</small></div>
        <div className="graph-annotation annotation-top"><span className="annotation-rule" />gated state update</div>
        <div className="graph-annotation annotation-bottom"><span className="annotation-rule amber-rule" />softmax projection</div>
      </div>
      <div className="graph-footer"><div><span className="graph-footer-label">RESIDUAL STREAM</span><span className="graph-footer-value">hidden state / layer {selectedLayer.replace('L', '')}</span></div><div className="graph-footer-right"><span>ACTIVE TOKEN</span><strong>{String(activeToken + 1).padStart(2, '0')} · {TOKENS[activeToken]}</strong></div></div>
    </section>
  )
}

function NeuronField({ step }) {
  const bars = Array.from({ length: 44 }, (_, index) => ((Math.sin(index * 1.7 + step * 0.035) + 1) / 2) * 0.75 + 0.08)
  return <div className="neuron-field" aria-label="Activation field">
    {bars.map((value, index) => <span key={index} className={index % 9 === 0 ? 'hot' : ''} style={{ height: `${value * 100}%` }} />)}
  </div>
}

function AttentionMap({ activeToken, values }) {
  const liveValues = Array.isArray(values) && values.length > 0 ? values.map(Number) : null
  const livePeak = liveValues ? Math.max(...liveValues, 0.0001) : 1
  return <div className="attention-map">{Array.from({ length: 49 }, (_, index) => {
    const row = Math.floor(index / 7)
    const column = index % 7
    const fallback = Math.max(0.06, 0.12 + Math.sin((row + 1) * (column + 2) + activeToken) * 0.11 + (row === activeToken ? 0.28 : 0))
    const liveIntensity = liveValues ? Math.min(0.88, 0.08 + ((liveValues[index % liveValues.length] || 0) / livePeak) * 0.8) : fallback
    const focus = row === activeToken || column === activeToken
    const intensity = focus ? Math.min(0.96, liveIntensity + 0.12) : liveIntensity
    return <span key={index} style={{ backgroundColor: `rgba(120, 228, 232, ${intensity})` }} />
  })}</div>
}

function Metric({ label, value }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong></div>
}

function WeightsView({ onSelect }) {
  return <section className="weights-panel"><div className="weights-header"><div><div className="crumb"><span>TRACE</span><ArrowDown size={12} /><span>WEIGHTS</span></div><h2>Parameter surface</h2><p>Weight-level inspection is reserved for the full-precision checkpoint.</p></div><button className="outline-button" onClick={onSelect}><SlidersHorizontal size={14} />View requirement</button></div><div className="weights-empty"><Sparkles size={22} /><span>Use BF16 safetensors for faithful gradients</span><small>The current runtime can still expose activation flow and token-level events.</small></div></section>
}

function TraceTransport({ isPlaying, setIsPlaying, speed, setSpeed, progress, step, setStep, processLabel }) {
  return <div className="transport"><button className="transport-play" onClick={() => setIsPlaying((current) => !current)} aria-label={isPlaying ? 'Pause trace' : 'Play trace'}>{isPlaying ? <Pause size={15} fill="currentColor" /> : <Play size={15} fill="currentColor" />}</button><div className="transport-meta"><span className="transport-label">{processLabel}</span><strong>Step {step} / 512</strong></div><input className="scrubber" type="range" min="0" max="512" value={step} onChange={(event) => setStep(Number(event.target.value))} style={{ '--progress': `${progress}%` }} aria-label="Trace step" /><div className="speed-control"><TimerReset size={14} /><select value={speed} onChange={(event) => setSpeed(Number(event.target.value))} aria-label="Playback speed"><option value="0.5">0.5×</option><option value="1">1×</option><option value="2">2×</option><option value="4">4×</option></select></div><span className="transport-state"><span className="transport-state-dot" />{isPlaying ? 'LIVE' : 'PAUSED'}</span></div>
}

export default App
