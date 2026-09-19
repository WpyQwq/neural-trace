import { useMemo, useRef, useState } from 'react'

const DEFAULT_LAYER_COUNT = 32
const NEURON_COUNT = 6

const layerLabel = (index) => `L${String(index + 1).padStart(2, '0')}`

function layerPosition(index, layerCount = DEFAULT_LAYER_COUNT) {
  const progress = index / Math.max(1, layerCount - 1)
  return [
    (progress - 0.5) * 15.4,
    Math.sin(progress * Math.PI * 3.2) * 0.68,
    Math.cos(progress * Math.PI * 2.4) * 0.72,
  ]
}

function projectPoint(point, yaw, pitch, zoom = 1) {
  const [x, y, z] = point
  const yawCos = Math.cos(yaw)
  const yawSin = Math.sin(yaw)
  const rotatedX = x * yawCos - z * yawSin
  const rotatedZ = x * yawSin + z * yawCos
  const pitchCos = Math.cos(pitch)
  const pitchSin = Math.sin(pitch)
  const rotatedY = y * pitchCos - rotatedZ * pitchSin
  const depth = y * pitchSin + rotatedZ * pitchCos
  const perspective = 1 / (1 + depth * 0.047)
  return {
    x: 500 + rotatedX * 47 * perspective * zoom,
    y: 230 - rotatedY * 56 * perspective * zoom,
    depth,
    perspective,
  }
}

function pointString(points) {
  return points.map((point) => `${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(' ')
}

function makeLayers(step, selectedLayer, yaw, pitch, signalValue, layerCount, zoom) {
  const activeIndex = step % layerCount
  const liveSignal = Number.isFinite(Number(signalValue)) ? Math.min(1, Math.max(0, Number(signalValue))) : null
  return Array.from({ length: layerCount }, (_, index) => {
    const label = layerLabel(index)
    const fullAttention = (index + 1) % 4 === 0
    const base = layerPosition(index, layerCount)
    const simulatedEnergy = 0.2 + ((Math.sin(index * 1.74 + step * 0.11) + 1) / 2) * 0.8
    const energy = liveSignal == null ? simulatedEnergy : Math.min(1, 0.16 + liveSignal * 0.72 + ((Math.sin(index * 1.74 + step * 0.11) + 1) / 2) * 0.18)
    const node = projectPoint(base, yaw, pitch, zoom)
    const neurons = Array.from({ length: NEURON_COUNT }, (_, neuronIndex) => {
      const angle = (neuronIndex / NEURON_COUNT) * Math.PI * 2
      const radius = 0.26 + energy * 0.05
      return projectPoint([base[0] + Math.cos(angle) * radius, base[1] + Math.sin(angle) * radius, base[2] + Math.sin(angle * 1.8) * 0.08], yaw, pitch, zoom)
    })
    return { index, label, fullAttention, energy, node, neurons, active: index === activeIndex, selected: selectedLayer === label }
  })
}

function FlowPacket({ step, yaw, pitch, mode, layerCount, zoom }) {
  const reverse = mode === 'backward'
  const currentIndex = reverse ? layerCount - 1 - (step % layerCount) : step % layerCount
  const nextIndex = reverse ? Math.max(0, currentIndex - 1) : Math.min(layerCount - 1, currentIndex + 1)
  const progress = (step % 16) / 16
  const current = layerPosition(currentIndex, layerCount)
  const next = layerPosition(nextIndex, layerCount)
  const point = current.map((value, index) => value + (next[index] - value) * progress)
  const projected = projectPoint([point[0], point[1] + 0.14, point[2]], yaw, pitch, zoom)
  return <circle cx={projected.x} cy={projected.y} r={4.4 * projected.perspective} fill={reverse ? '#efb56b' : '#78e4e8'} className="svg-packet" />
}

function ProjectionGrid() {
  return <g className="projection-grid">
    {Array.from({ length: 13 }, (_, index) => {
      const x = 40 + index * 77
      return <line key={`v-${index}`} x1={x} y1="58" x2={x} y2="407" />
    })}
    {Array.from({ length: 7 }, (_, index) => {
      const y = 70 + index * 54
      return <line key={`h-${index}`} x1="30" y1={y} x2="970" y2={y} />
    })}
  </g>
}

export default function Trace3DCanvas({ layerCount = DEFAULT_LAYER_COUNT, pattern = '8 × (3L + 1A)', hasImage = false, signalKind = 'waiting', step, selectedLayer, setSelectedLayer, activeToken, mode = 'forward', signalValue = null }) {
  const [orbit, setOrbit] = useState({ yaw: -0.12, pitch: 0.13 })
  const [zoom, setZoom] = useState(1)
  const dragRef = useRef(null)
  const safeLayerCount = Math.max(1, Number(layerCount) || DEFAULT_LAYER_COUNT)
  const layers = useMemo(() => makeLayers(step, selectedLayer, orbit.yaw, orbit.pitch, signalValue, safeLayerCount, zoom), [orbit.pitch, orbit.yaw, safeLayerCount, selectedLayer, signalValue, step, zoom])
  const sortedLayers = useMemo(() => [...layers].sort((a, b) => a.node.depth - b.node.depth), [layers])
  const residualPoints = useMemo(() => layers.map((layer) => layer.node), [layers])
  const attentionLinks = useMemo(() => layers.filter((layer) => layer.fullAttention).map((layer) => {
    const next = layers[Math.min(layers.length - 1, layer.index + 1)]
    return { id: layer.label, points: [layer.node, next.node] }
  }), [layers])
  const auxLayer = layers.find((layer) => layer.selected) || layers[0]
  const auxLabel = signalKind === 'router_weights' ? 'ROUTER' : signalKind === 'expert_mixture' ? 'EXPERT MIX' : signalKind === 'mtp_logits' ? 'MTP' : ''

  const handlePointerDown = (event) => {
    event.currentTarget.setPointerCapture(event.pointerId)
    dragRef.current = { x: event.clientX, y: event.clientY, ...orbit }
  }

  const handlePointerMove = (event) => {
    if (!dragRef.current) return
    setOrbit({
      yaw: dragRef.current.yaw + (event.clientX - dragRef.current.x) * 0.007,
      pitch: Math.max(-0.45, Math.min(0.45, dragRef.current.pitch + (event.clientY - dragRef.current.y) * 0.004)),
    })
  }

  const stopDragging = () => { dragRef.current = null }
  const handleWheel = (event) => {
    event.preventDefault()
    setZoom((current) => Math.max(0.72, Math.min(1.55, current - event.deltaY * 0.001)))
  }

  return <section className="three-d-panel">
    <div className="three-d-toolbar">
      <div className="three-d-title"><span className="three-d-mark" /><div><strong>3D TRACE SPACE</strong><small>observable compute · step {step} / 512</small></div></div>
      <div className="three-d-toolbar-meta"><span><i className="legend-swatch cyan" />DeltaNet state</span><span><i className="legend-swatch amber" />full attention</span>{hasImage && <span><i className="legend-swatch amber" />vision input</span>}<span>token {String(activeToken + 1).padStart(2, '0')}</span></div>
    </div>
    <div className="three-d-viewport svg-viewport" onPointerDown={handlePointerDown} onPointerMove={handlePointerMove} onPointerUp={stopDragging} onPointerCancel={stopDragging} onPointerLeave={stopDragging} onWheel={handleWheel}>
      <svg viewBox="0 0 1000 460" role="img" aria-label="3D projected model architecture">
        <defs>
          <filter id="node-glow" x="-100%" y="-100%" width="300%" height="300%"><feGaussianBlur stdDeviation="3" result="blur" /><feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
        </defs>
        <rect width="1000" height="460" fill="#0b0d10" />
        <ProjectionGrid />
        {hasImage && <g className="svg-vision-branch"><line x1="76" y1="365" x2={layers[0].node.x} y2={layers[0].node.y} /><circle cx="76" cy="365" r="6" /><text x="58" y="386">IMAGE</text></g>}
        <polyline points={pointString(residualPoints)} className={`svg-residual ${mode === 'state' ? 'is-emphasis' : ''} ${mode === 'backward' ? 'is-backward' : ''}`} />
        {attentionLinks.map((link) => <line key={link.id} x1={link.points[0].x} y1={link.points[0].y} x2={link.points[1].x} y2={link.points[1].y} className={`svg-attention ${mode === 'attention' ? 'is-emphasis' : ''}`} />)}
        {auxLabel && auxLayer && <g className="svg-aux-branch"><line x1={auxLayer.node.x} y1={auxLayer.node.y} x2={auxLayer.node.x + 58} y2={auxLayer.node.y - 34} /><circle cx={auxLayer.node.x + 58} cy={auxLayer.node.y - 34} r="4" /><text x={auxLayer.node.x + 66} y={auxLayer.node.y - 31}>{auxLabel}</text></g>}
        {sortedLayers.map((layer) => <g key={layer.label} role="button" tabIndex="0" aria-label={`Inspect ${layer.label}`} className={`svg-layer ${layer.selected ? 'is-selected' : ''} ${layer.active ? 'is-active' : ''}`} onClick={(event) => { event.stopPropagation(); setSelectedLayer(layer.label) }} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); setSelectedLayer(layer.label) } }}>
          {layer.neurons.map((neuron, index) => <circle key={index} cx={neuron.x} cy={neuron.y} r={layer.active ? 2.5 : 1.8} fill={layer.fullAttention ? '#efb56b' : '#78e4e8'} opacity={(0.18 + layer.energy * 0.45) * neuron.perspective} />)}
          {layer.selected && <circle cx={layer.node.x} cy={layer.node.y} r={12 * layer.node.perspective} className="svg-selection" />}
          <circle cx={layer.node.x} cy={layer.node.y} r={(layer.active ? 7.6 : 5.2) * layer.node.perspective} fill={layer.fullAttention ? '#efb56b' : '#78e4e8'} opacity={0.98} filter={layer.active ? 'url(#node-glow)' : undefined} />
          <text x={layer.node.x} y={layer.node.y - 14 * layer.node.perspective} className="svg-layer-label" opacity={0.45 + layer.node.perspective * 0.55}>{layer.label}</text>
        </g>)}
        <FlowPacket step={step} yaw={orbit.yaw} pitch={orbit.pitch} mode={mode} layerCount={safeLayerCount} zoom={zoom} />
        <g className="svg-axes"><line x1="48" y1="416" x2="127" y2="416" /><line x1="48" y1="416" x2="48" y2="337" /><text x="132" y="420">X</text><text x="43" y="330">Y</text><text x="35" y="438">Z / depth</text></g>
      </svg>
      <div className="three-d-axis axis-x">RESIDUAL STREAM <span>→</span></div>
      <div className="three-d-axis axis-y">HIDDEN STATE <span>↕</span></div>
      <div className="three-d-readout"><span>ACTIVE LAYER</span><strong>{selectedLayer}</strong><small>click a node to inspect</small></div>
      <div className="three-d-hint">DRAG TO ORBIT&nbsp;&nbsp;·&nbsp;&nbsp;SCROLL TO ZOOM</div>
    </div>
    <div className="three-d-footer"><span>{safeLayerCount} LAYERS · {pattern}</span><span>{hasImage ? 'TELEMETRY: VISION / ACTIVATION / STATE / LOGITS' : 'TELEMETRY: ACTIVATION / STATE / LOGITS'}</span><span>PRECISION: BF16 / INT4</span></div>
  </section>
}
