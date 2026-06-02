/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef, useEffect, useMemo } from 'react';
import axios from 'axios';
import { 
  Upload, Sun, ShieldAlert, CheckCircle2, History as HistoryIcon, 
  AlertTriangle, Info, LayoutDashboard, BarChart3, Clock, Calendar, 
  Trash2, FileText, ChevronRight, Activity, Cpu, Eye, Compass, X, Download
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
} from 'recharts';

// --- Types ---

interface Detection {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: {
    x_min: number;
    y_min: number;
    x_max: number;
    y_max: number;
    width: number;
    height: number;
  };
}

interface ApiResponse {
  status: string;
  message: string;
  detections: Detection[];
}

interface HistoryItem {
  id: string;
  filename: string;
  date: string;
  time: string;
  day: string;
  detections: Detection[];
}

type View = 'upload' | 'history' | 'stats';

// --- Constants ---

const COLORS = [
  '#f97316', // Hotspot Orange
  '#ef4444', // Infrared Red
  '#eab308', // Amber
  '#6366f1', // IA Indigo
  '#06b6d4', // Cyan
  '#8b5cf6', // Purple
  '#10b981', // System Green
  '#ec4899'  // Rose
];

const CLASS_NAMES_LIST = [
  "Single Hotspot", "Multi Hotspots", "Single Diode", "Multi Diode",
  "Single Bypassed Substring", "Multi Bypassed Substring", "String Open Circuit", "String Reversed Polarity"
];

// --- Helper Functions ---

const formatDate = (date: Date) => {
  const days = ['DOMINGO', 'LUNES', 'MARTES', 'MIÉRCOLES', 'JUEVES', 'VIERNES', 'SÁBADO'];
  return {
    date: date.toLocaleDateString('es-ES', { year: 'numeric', month: '2-digit', day: '2-digit' }),
    time: date.toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
    day: days[date.getDay()]
  };
};

export default function App() {
  const [currentView, setCurrentView] = useState<View>('upload');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<ApiResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [hoveredDetectionIndex, setHoveredDetectionIndex] = useState<number | null>(null);
  const [selectedHistoryItem, setSelectedHistoryItem] = useState<HistoryItem | null>(null);
  const [mouseCoords, setMouseCoords] = useState<{ x: number; y: number } | null>(null);
  const [dragActive, setDragActive] = useState(false);
  
  // Custom tooltips state (Emil Kowalski - skip delay on adjacent tooltips)
  const [hoveredTooltip, setHoveredTooltip] = useState<string | null>(null);
  const [anyTooltipOpen, setAnyTooltipOpen] = useState(false);
  const tooltipTimeoutRef = useRef<number | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const imageContainerRef = useRef<HTMLDivElement>(null);

  // Load history from localStorage on mount
  useEffect(() => {
    const savedHistory = localStorage.getItem('solar_history');
    if (savedHistory) {
      try {
        setHistory(JSON.parse(savedHistory));
      } catch (e) {
        console.error('Failed to load history', e);
      }
    }
  }, []);

  // Save history to localStorage when it changes
  useEffect(() => {
    localStorage.setItem('solar_history', JSON.stringify(history));
  }, [history]);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setResults(null);
      setError(null);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('image/')) {
        setSelectedFile(file);
        const url = URL.createObjectURL(file);
        setPreviewUrl(url);
        setResults(null);
        setError(null);
      } else {
        setError('El archivo debe ser una imagen (JPG, PNG).');
      }
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post<ApiResponse>('https://web-production-4963b.up.railway.app/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      const newResults = response.data;
      setResults(newResults);

      // Add to history
      const { date, time, day } = formatDate(new Date());
      const newHistoryItem: HistoryItem = {
        id: crypto.randomUUID(),
        filename: selectedFile.name,
        date,
        time,
        day,
        detections: newResults.detections
      };

      setHistory(prev => [newHistoryItem, ...prev]);

    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.detail || 'Error al conectar con la API de IA. Revisa la conectividad del servidor.');
    } finally {
      setLoading(false);
    }
  };

  const deleteHistoryItem = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (window.confirm('¿Eliminar este registro de inspección?')) {
      setHistory(prev => prev.filter(item => item.id !== id));
      if (selectedHistoryItem?.id === id) {
        setSelectedHistoryItem(null);
      }
    }
  };

  const clearHistory = () => {
    if (window.confirm('¿Estás seguro de que quieres borrar todo el historial?')) {
      setHistory([]);
      setSelectedHistoryItem(null);
    }
  };

  // Tracking coords inside thermal viewport
  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!imageContainerRef.current) return;
    const rect = imageContainerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(640, Math.round(((e.clientX - rect.left) / rect.width) * 640)));
    const y = Math.max(0, Math.min(640, Math.round(((e.clientY - rect.top) / rect.height) * 640)));
    setMouseCoords({ x, y });
  };

  // Tooltip triggers (Emil Kowalski rule)
  const showTooltip = (id: string) => {
    if (tooltipTimeoutRef.current) clearTimeout(tooltipTimeoutRef.current);
    setHoveredTooltip(id);
    setAnyTooltipOpen(true);
  };

  const hideTooltip = () => {
    tooltipTimeoutRef.current = window.setTimeout(() => {
      setHoveredTooltip(null);
      setAnyTooltipOpen(false);
    }, 300);
  };

  // --- Computed Stats ---
  const statsData = useMemo(() => {
    const counts: Record<string, number> = {};
    CLASS_NAMES_LIST.forEach(name => (counts[name] = 0));
    
    let totalConfidence = 0;
    let detectionsCount = 0;
    
    history.forEach(item => {
      item.detections.forEach(d => {
        counts[d.class_name] = (counts[d.class_name] || 0) + 1;
        totalConfidence += d.confidence;
        detectionsCount++;
      });
    });

    const total = Object.values(counts).reduce((acc, curr) => acc + curr, 0);

    const barData = Object.entries(counts).map(([name, count]) => ({
      name,
      count,
      percentage: total > 0 ? parseFloat(((count / total) * 100).toFixed(1)) : 0
    })).filter(d => d.count > 0);

    const avgConfidence = detectionsCount > 0 ? (totalConfidence / detectionsCount) * 100 : 0;
    
    // Procedural Solar Panel Efficiency: starts at 100%, drops by 1.5% for every single anomaly detected
    const solarEfficiency = Math.max(70, 100 - (total * 1.5));

    return { barData, total, avgConfidence, solarEfficiency };
  }, [history]);

  // Procedural Thermal Reconstruction component for History Drawer
  const ThermalReconstruction = ({ detections }: { detections: Detection[] }) => {
    return (
      <div className="relative w-full aspect-video bg-[#09090b] border border-white/5 rounded-xl overflow-hidden flex items-center justify-center">
        {/* Background Solar Cell Grid Layout */}
        <svg className="absolute inset-0 w-full h-full opacity-[0.06]" viewBox="0 0 120 60" preserveAspectRatio="none">
          <defs>
            <pattern id="grid-cell" width="10" height="10" patternUnits="userSpaceOnUse">
              <rect width="9" height="9" fill="none" stroke="#06b6d4" strokeWidth="0.5" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid-cell)" />
        </svg>

        {/* Dynamic Heat Gradient Maps */}
        <svg className="absolute inset-0 w-full h-full" viewBox="0 0 640 640">
          <defs>
            {detections.map((d, i) => (
              <radialGradient key={i} id={`hist-heat-${i}`} cx="50%" cy="50%" r="50%">
                <stop offset="0%" stopColor="#ef4444" stopOpacity="0.85" />
                <stop offset="35%" stopColor="#f97316" stopOpacity="0.5" />
                <stop offset="75%" stopColor="#eab308" stopOpacity="0.15" />
                <stop offset="100%" stopColor="#ef4444" stopOpacity="0" />
              </radialGradient>
            ))}
          </defs>
          {detections.map((d, i) => {
            const cx = d.bbox.x_min + d.bbox.width / 2;
            const cy = d.bbox.y_min + d.bbox.height / 2;
            const r = Math.max(40, Math.max(d.bbox.width, d.bbox.height) * 0.7);
            return (
              <circle
                key={i}
                cx={cx}
                cy={cy}
                r={r}
                fill={`url(#hist-heat-${i})`}
                className="animate-pulse"
                style={{ animationDuration: `${1.5 + (i % 2)}s` }}
              />
            );
          })}
        </svg>

        {/* Matrix HUD Overlays */}
        <div className="absolute top-3 left-3 text-[9px] font-mono text-cyan-500/50 uppercase tracking-widest bg-cyan-950/20 px-2 py-0.5 rounded border border-cyan-500/10">
          Reconst_Thermal_Matrix_v9
        </div>

        {detections.length === 0 ? (
          <div className="text-center z-10 p-6 bg-zinc-950/50 backdrop-blur-md border border-[#10b981]/20 rounded-xl max-w-xs">
            <CheckCircle2 className="w-8 h-8 text-[#10b981] mx-auto mb-2" />
            <p className="text-xs font-mono text-[#10b981] uppercase tracking-widest font-bold">Diagnóstico Limpio</p>
            <p className="text-[10px] text-gray-500 mt-1">Inspección completada con cero anomalías térmicas en este nodo.</p>
          </div>
        ) : (
          <div className="absolute bottom-3 right-3 text-[9px] font-mono text-orange-500/80 bg-orange-950/20 px-2 py-0.5 rounded border border-orange-500/10">
            {detections.length} Anomalías Registradas
          </div>
        )}
      </div>
    );
  };

  // --- Views ---

  const UploadView = () => (
    <div className="grid lg:grid-cols-12 gap-6 items-start">
      {/* Control Panel */}
      <div className="lg:col-span-4 space-y-6">
        <section id="upload-section" className="glass-panel rounded-2xl p-6 glass-panel-hover">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-telemetry-orange/10 rounded-lg text-telemetry-orange">
              <Upload className="w-5 h-5" />
            </div>
            <div>
              <h2 className="font-bold text-sm uppercase tracking-wider text-white">Ingesta de Telemetría</h2>
              <p className="text-[10px] text-gray-500 font-mono">STANDBY_WAITING_FOR_DATA</p>
            </div>
          </div>

          <div 
            id="drop-zone"
            onClick={() => fileInputRef.current?.click()}
            onDragEnter={handleDrag}
            onDragOver={handleDrag}
            onDragLeave={handleDrag}
            onDrop={handleDrop}
            className={`
              border border-dashed rounded-xl p-8 transition-all duration-300 cursor-pointer group text-center
              ${dragActive 
                ? 'border-telemetry-orange bg-telemetry-orange/5 shadow-[0_0_20px_rgba(249,115,22,0.15)]' 
                : selectedFile 
                  ? 'border-telemetry-orange/30 bg-telemetry-orange/5' 
                  : 'border-zinc-800 hover:border-zinc-700 bg-zinc-950/45'}
            `}
          >
            <input 
              type="file" 
              ref={fileInputRef}
              onChange={handleFileSelect}
              className="hidden" 
              accept="image/*"
            />
            <div className="flex flex-col items-center">
              <div className={`p-4 rounded-full mb-4 transition-all duration-300 active-press ${selectedFile ? 'bg-telemetry-orange text-white shadow-[0_0_15px_rgba(249,115,22,0.4)]' : 'bg-zinc-900 text-zinc-500 group-hover:text-zinc-300'}`}>
                <Upload className="w-6 h-6" />
              </div>
              <p className="font-medium mb-1 text-xs tracking-wide text-zinc-200">
                {selectedFile ? selectedFile.name : 'Arrastra o selecciona la termografía'}
              </p>
              <p className="text-[9px] text-zinc-500 uppercase tracking-widest font-mono">Formatos JPG, PNG (Max 10MB)</p>
            </div>
          </div>

          <button
            id="analyze-button"
            disabled={!selectedFile || loading}
            onClick={handleUpload}
            className={`
              w-full mt-5 py-3.5 rounded-xl font-bold flex items-center justify-center gap-3 transition-all duration-300 active-press
              ${!selectedFile || loading 
                ? 'bg-zinc-900 border border-zinc-800 text-zinc-600 cursor-not-allowed' 
                : 'bg-telemetry-orange hover:bg-orange-600 text-white font-bold tracking-wider shadow-[0_0_25px_rgba(249,115,22,0.25)] border border-orange-500/20'}
            `}
          >
            {loading ? (
              <div className="w-5 h-5 border-2 border-white/20 border-t-white rounded-full animate-spin" />
            ) : (
              <Cpu className="w-5 h-5" />
            )}
            <span className="font-mono text-xs uppercase tracking-wider">
              {loading ? 'Procesando Red Neuronal...' : 'Ejecutar Diagnóstico IA'}
            </span>
          </button>
        </section>

        {/* Engine Telemetry Info */}
        <section id="info-section" className="glass-panel rounded-2xl p-6 text-zinc-400">
          <div className="flex items-center gap-2 mb-4">
            <Info className="w-4 h-4 text-zinc-500" />
            <h2 className="font-bold text-xs uppercase tracking-widest text-zinc-300">Motor SolarGuard IA</h2>
          </div>
          <div className="space-y-2.5 font-mono text-[11px]">
            <div className="flex justify-between py-1.5 border-b border-zinc-900/60">
              <span className="text-zinc-500">Arquitectura</span>
              <span className="text-zinc-300">YOLOv9-Small (FineTuned)</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-zinc-900/60">
              <span className="text-zinc-500">Resolución</span>
              <span className="text-zinc-300">640x640px</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-zinc-900/60">
              <span className="text-zinc-500">Precisión mAP</span>
              <span className="text-zinc-300 font-bold text-telemetry-cyan">94.2%</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-zinc-900/60">
              <span className="text-zinc-500">Clases Clínicas</span>
              <span className="text-zinc-300">8 Anomalías</span>
            </div>
            <div className="flex justify-between py-1.5">
              <span className="text-zinc-500">Tipo Inspección</span>
              <span className="text-zinc-300">Fotovoltaica Aérea</span>
            </div>
          </div>
        </section>
      </div>

      {/* Interactive Thermal Viewport */}
      <div className="lg:col-span-8 flex flex-col gap-6">
        <section id="view-section" className="glass-panel rounded-2xl overflow-hidden flex-1 relative flex flex-col">
          <div className="bg-[#0b0b0e]/90 border-b border-white/5 px-6 py-4 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-telemetry-red animate-pulse" />
              <span className="text-[10px] font-mono text-zinc-400 uppercase tracking-widest">Viewport de Telemetría Térmica</span>
            </div>
            
            {results && (
              <span className="bg-telemetry-orange/10 text-telemetry-orange text-[9px] px-2.5 py-0.5 rounded font-mono border border-telemetry-orange/20 tracking-wider">
                NODE_DIAGNOSED_OK
              </span>
            )}
          </div>

          <div className="p-6 flex-1 flex flex-col items-center justify-center min-h-[350px] relative">
            {!previewUrl ? (
              <div className="text-center text-zinc-600 max-w-md py-12">
                <Compass className="w-16 h-16 mx-auto mb-4 text-zinc-700 animate-spin" style={{ animationDuration: '20s' }} />
                <p className="text-xs font-mono uppercase tracking-widest text-zinc-500">Modo Telemetría en Standby</p>
                <p className="text-[10px] text-zinc-600 mt-2 font-mono">Esperando flujo de imágenes termográficas para inicializar el análisis fotovoltaico...</p>
              </div>
            ) : (
              <div 
                ref={imageContainerRef}
                onMouseMove={handleMouseMove}
                onMouseLeave={() => setMouseCoords(null)}
                className="relative max-w-full rounded-xl overflow-hidden shadow-2xl border border-white/5 cursor-crosshair group select-none"
              >
                <img 
                  src={previewUrl} 
                  className="max-h-[50vh] object-contain block transition-all duration-500" 
                  alt="Termografía" 
                />
                
                {/* HUD Crosshairs Following Cursor */}
                {mouseCoords && (
                  <>
                    <div className="absolute left-0 right-0 h-[0.5px] bg-telemetry-orange/30 pointer-events-none" style={{ top: `${(mouseCoords.y / 640) * 100}%` }} />
                    <div className="absolute top-0 bottom-0 w-[0.5px] bg-telemetry-orange/30 pointer-events-none" style={{ left: `${(mouseCoords.x / 640) * 100}%` }} />
                    <div className="absolute text-[8px] font-mono text-telemetry-orange bg-zinc-950/80 px-1 py-0.5 border border-telemetry-orange/20 rounded pointer-events-none shadow" style={{ left: `${(mouseCoords.x / 640) * 100 + 2}%`, top: `${(mouseCoords.y / 640) * 100 + 2}%` }}>
                      X:{mouseCoords.x} Y:{mouseCoords.y}
                    </div>
                  </>
                )}

                {/* Laser scan line overlay when loading (AI inference in progress) */}
                {loading && (
                  <div className="absolute inset-x-0 h-1 bg-gradient-to-r from-transparent via-telemetry-orange to-transparent shadow-[0_0_15px_rgba(249,115,22,0.8)] animate-laser-scan pointer-events-none" />
                )}
                
                {/* Interactive Bounding Boxes Overlay */}
                {results && results.detections.map((d, index) => {
                  const isHovered = hoveredDetectionIndex === index;
                  return (
                    <div 
                      key={index}
                      onMouseEnter={() => setHoveredDetectionIndex(index)}
                      onMouseLeave={() => setHoveredDetectionIndex(null)}
                      className={`
                        absolute border-2 pointer-events-auto transition-all duration-150 cursor-pointer
                        ${isHovered 
                          ? 'border-telemetry-red bg-telemetry-red/10 z-30 shadow-[0_0_15px_rgba(239,68,68,0.4)] scale-[1.01]' 
                          : 'border-telemetry-orange bg-telemetry-orange/5 z-20'}
                      `}
                      style={{
                        left: `${(d.bbox.x_min / 640) * 100}%`,
                        top: `${(d.bbox.y_min / 640) * 100}%`,
                        width: `${(d.bbox.width / 640) * 100}%`,
                        height: `${(d.bbox.height / 640) * 100}%`,
                      }}
                    >
                      <span className={`
                        absolute -top-5 left-[-2px] text-[8px] font-mono uppercase font-bold whitespace-nowrap px-1.5 py-0.5 rounded shadow-lg transition-colors
                        ${isHovered ? 'bg-telemetry-red text-white' : 'bg-telemetry-orange text-white'}
                      `}>
                        {d.class_name} | {(d.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </section>

        {/* Results Summary with Hover Interactivity */}
        <section id="analysis-results" className="glass-panel rounded-2xl p-6 min-h-[160px] flex flex-col justify-center">
          <AnimatePresence mode="wait">
            {results ? (
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.2 }}
                key="results"
                className="space-y-4"
              >
                <div className="flex items-center justify-between border-b border-zinc-900 pb-3 font-mono">
                  <h3 className="text-xs uppercase tracking-widest text-zinc-500 flex items-center gap-2">
                    <Activity className="w-4 h-4 text-telemetry-indigo" /> Hallazgos Clínicos
                  </h3>
                  <span className="text-[10px] text-zinc-400 bg-zinc-900 px-2 py-0.5 rounded">
                    {results.detections.length} Fibras / Celdas Afectadas
                  </span>
                </div>
                
                <div className="grid sm:grid-cols-2 gap-3">
                  {results.detections.length > 0 ? (
                    results.detections.map((d, i) => {
                      const isHovered = hoveredDetectionIndex === i;
                      return (
                        <div 
                          key={i} 
                          onMouseEnter={() => setHoveredDetectionIndex(i)}
                          onMouseLeave={() => setHoveredDetectionIndex(null)}
                          className={`
                            border rounded-xl p-3.5 flex items-center gap-4 transition-all duration-200 cursor-pointer
                            ${isHovered 
                              ? 'border-telemetry-red/40 bg-telemetry-red/5 shadow-[0_4px_12px_rgba(239,68,68,0.05)]' 
                              : 'border-zinc-800 bg-zinc-950/20 hover:border-zinc-700'}
                          `}
                        >
                          <div className={`
                            p-2 rounded-lg transition-all duration-300
                            ${isHovered 
                              ? 'bg-telemetry-red/20 text-telemetry-red' 
                              : 'bg-telemetry-orange/10 text-telemetry-orange'}
                          `}>
                            <AlertTriangle className="w-4 h-4" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="text-xs font-semibold text-zinc-200 truncate">{d.class_name}</p>
                            <p className="text-[9px] text-zinc-500 font-mono tracking-tighter uppercase mt-0.5">
                              Confianza: {(d.confidence * 100).toFixed(1)}% | Area: {Math.round(d.bbox.width * d.bbox.height)}px²
                            </p>
                          </div>
                        </div>
                      );
                    })
                  ) : (
                    <div className="col-span-2 py-6 flex flex-col items-center justify-center">
                      <CheckCircle2 className="w-10 h-10 mb-3 text-telemetry-green animate-pulse" />
                      <p className="text-xs font-mono uppercase tracking-widest text-[#10b981] font-bold">Firma Térmica Nominal</p>
                      <p className="text-[10px] text-zinc-500 mt-1">No se detectan anomalías de temperatura en las celdas del panel.</p>
                    </div>
                  )}
                </div>
              </motion.div>
            ) : error ? (
              <motion.div 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                key="error"
                className="p-4 bg-telemetry-red/10 border border-telemetry-red/20 rounded-xl text-telemetry-red flex items-start gap-3 font-mono"
              >
                <AlertTriangle className="w-5 h-5 mt-0.5 flex-shrink-0" />
                <div className="space-y-1">
                  <p className="text-xs font-bold uppercase tracking-widest">Error del Nodo de Procesamiento</p>
                  <p className="text-[10px] opacity-80 leading-relaxed">{error}</p>
                </div>
              </motion.div>
            ) : (
              <div className="flex flex-col items-center justify-center py-6 opacity-30">
                <Clock className="w-6 h-6 mb-2 text-zinc-500" />
                <p className="text-[10px] font-mono uppercase tracking-[0.2em] text-zinc-400">Espera de Inferencia de Telemetría</p>
              </div>
            )}
          </AnimatePresence>
        </section>
      </div>
    </div>
  );

  const HistoryView = () => (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
      <div className="flex items-center justify-between border-b border-zinc-900 pb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-telemetry-orange/10 rounded-lg text-telemetry-orange">
            <HistoryIcon className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-white uppercase tracking-wider">Histórico de Inspección</h2>
            <p className="text-[10px] text-zinc-500 font-mono">REGISTRY_DATABASE_ONLINE</p>
          </div>
        </div>
        <button 
          onClick={clearHistory}
          disabled={history.length === 0}
          className="flex items-center gap-2 px-4 py-2 bg-telemetry-red/10 text-telemetry-red border border-telemetry-red/20 rounded-xl hover:bg-telemetry-red hover:text-white transition-all duration-300 disabled:opacity-30 disabled:cursor-not-allowed text-xs font-mono uppercase tracking-wider active-press"
        >
          <Trash2 className="w-4 h-4" /> Borrar Base de Datos
        </button>
      </div>

      <div className="glass-panel rounded-2xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-zinc-900 bg-zinc-950/65 text-zinc-500 text-[10px] uppercase font-mono tracking-widest">
                <th className="px-6 py-4.5 font-medium">Archivo Termográfico</th>
                <th className="px-6 py-4.5 font-medium">Fecha de Registro</th>
                <th className="px-6 py-4.5 font-medium">Hora Local</th>
                <th className="px-6 py-4.5 font-medium">Diagnóstico IA</th>
                <th className="px-6 py-4.5 font-medium text-right">Telemetría</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-900 font-mono text-xs">
              {history.length > 0 ? (
                history.map((item, index) => (
                  <tr 
                    key={item.id} 
                    onClick={() => setSelectedHistoryItem(item)}
                    className="hover:bg-zinc-900/35 transition-colors cursor-pointer group stagger-item"
                    style={{ animationDelay: `${index * 30}ms` }}
                  >
                    <td className="px-6 py-4.5">
                      <div className="flex items-center gap-3">
                        <FileText className="w-4 h-4 text-zinc-600 group-hover:text-telemetry-orange transition-colors" />
                        <span className="font-semibold text-zinc-300 group-hover:text-white transition-colors">{item.filename}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4.5 text-zinc-400">
                      <div className="flex flex-col">
                        <span>{item.date}</span>
                        <span className="text-[9px] text-zinc-600 uppercase mt-0.5">{item.day}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4.5 text-zinc-500">
                      {item.time}
                    </td>
                    <td className="px-6 py-4.5">
                      <span className={`
                        inline-flex px-2.5 py-0.5 rounded text-[9px] font-bold border
                        ${item.detections.length > 0 
                          ? 'bg-telemetry-red/10 border-telemetry-red/20 text-telemetry-red' 
                          : 'bg-telemetry-green/10 border-telemetry-green/20 text-telemetry-green'}
                      `}>
                        {item.detections.length > 0 ? `${item.detections.length} ANOMALÍAS` : 'ESTABLE'}
                      </span>
                    </td>
                    <td className="px-6 py-4.5 text-right">
                      <button className="text-zinc-600 group-hover:text-zinc-300 transition-colors p-1 bg-zinc-950/40 border border-zinc-900 hover:border-zinc-800 rounded-lg">
                        <ChevronRight className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={5} className="px-6 py-20 text-center text-zinc-600">
                    <div className="flex flex-col items-center gap-3 opacity-30 py-8">
                      <HistoryIcon className="w-12 h-12 text-zinc-700" />
                      <p className="text-xs uppercase tracking-widest font-mono">Cero registros en memoria</p>
                      <p className="text-[10px]">Realiza un análisis en el módulo de diagnóstico para poblar la base de datos.</p>
                    </div>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* History Detail Drawer (Slide Over Panel) - Premium Emil Kowalski Component */}
      <AnimatePresence>
        {selectedHistoryItem && (
          <>
            {/* Backdrop */}
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 0.5 }}
              exit={{ opacity: 0 }}
              onClick={() => setSelectedHistoryItem(null)}
              className="fixed inset-0 bg-[#030304] z-50 pointer-events-auto"
            />
            {/* Drawer */}
            <motion.aside 
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              transition={{ type: 'spring', damping: 25, stiffness: 220 }}
              className="fixed right-0 top-0 bottom-0 w-full max-w-lg bg-[#0c0c0f] border-l border-zinc-800 z-50 shadow-[0_0_50px_rgba(0,0,0,0.8)] flex flex-col pointer-events-auto"
            >
              {/* Header */}
              <div className="p-6 border-b border-zinc-900 flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-bold text-white uppercase tracking-wider font-sans">Reporte Técnico de Telemetría</h3>
                  <p className="text-[9px] text-zinc-500 font-mono uppercase mt-0.5">{selectedHistoryItem.id}</p>
                </div>
                <button 
                  onClick={() => setSelectedHistoryItem(null)}
                  className="p-1.5 rounded-lg border border-zinc-800 text-zinc-500 hover:text-white bg-zinc-950/45 hover:bg-zinc-900 transition-colors active-press"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              {/* Content */}
              <div className="flex-1 overflow-y-auto p-6 space-y-6 font-mono">
                {/* Meta details */}
                <div className="grid grid-cols-2 gap-4 bg-zinc-950/60 p-4 border border-zinc-900 rounded-xl text-[10px]">
                  <div>
                    <span className="text-zinc-500 uppercase tracking-widest block">Archivo</span>
                    <span className="text-zinc-200 mt-1 block truncate font-semibold">{selectedHistoryItem.filename}</span>
                  </div>
                  <div>
                    <span className="text-zinc-500 uppercase tracking-widest block">Estampa Temporal</span>
                    <span className="text-zinc-200 mt-1 block font-semibold">{selectedHistoryItem.date} | {selectedHistoryItem.time}</span>
                  </div>
                </div>

                {/* Procedural Heat Map Reconstructor */}
                <div>
                  <h4 className="text-[10px] text-zinc-500 uppercase tracking-widest mb-2 flex items-center gap-1.5">
                    <Compass className="w-3.5 h-3.5 text-telemetry-cyan" /> Reconstitución de Telefirma
                  </h4>
                  <ThermalReconstruction detections={selectedHistoryItem.detections} />
                </div>

                {/* Anomaly breakdown */}
                <div className="space-y-3">
                  <h4 className="text-[10px] text-zinc-500 uppercase tracking-widest flex items-center gap-1.5">
                    <Eye className="w-3.5 h-3.5 text-telemetry-orange" /> Desglose de Celdas Afectadas
                  </h4>
                  {selectedHistoryItem.detections.length > 0 ? (
                    <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
                      {selectedHistoryItem.detections.map((d, i) => (
                        <div key={i} className="p-3 bg-[#0d0d10] border border-zinc-900 rounded-xl flex items-center justify-between text-[11px]">
                          <div className="flex items-center gap-3">
                            <span className="w-1.5 h-1.5 rounded-full bg-telemetry-red" />
                            <div>
                              <p className="font-semibold text-zinc-200">{d.class_name}</p>
                              <p className="text-[9px] text-zinc-500 mt-0.5">X_MIN:{d.bbox.x_min} | Y_MIN:{d.bbox.y_min} | W:{d.bbox.width} | H:{d.bbox.height}</p>
                            </div>
                          </div>
                          <span className="text-[10px] font-bold text-telemetry-orange">{(d.confidence * 100).toFixed(0)}% Confianza</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="p-4 border border-zinc-900 bg-zinc-950/20 rounded-xl text-center text-zinc-600 text-[10px]">
                      Ningún punto de calor anómalo fue detectado en esta termografía.
                    </div>
                  )}
                </div>
              </div>

              {/* Footer */}
              <div className="p-6 border-t border-zinc-900 bg-zinc-950/30 grid grid-cols-2 gap-4">
                <button 
                  onClick={() => alert('Exportando Reporte de Telemetría (PDF)...')}
                  className="flex items-center justify-center gap-2 py-3 border border-zinc-800 bg-zinc-950/50 hover:bg-zinc-900 text-zinc-300 text-xs rounded-xl font-mono uppercase tracking-wider transition-colors active-press"
                >
                  <Download className="w-4 h-4" /> Exportar Reporte
                </button>
                <button 
                  onClick={() => setSelectedHistoryItem(null)}
                  className="py-3 bg-zinc-900 hover:bg-zinc-800 text-zinc-200 text-xs rounded-xl font-mono uppercase tracking-wider transition-colors active-press"
                >
                  Cerrar
                </button>
              </div>
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    </motion.div>
  );

  const StatsView = () => (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
      <div className="flex items-center gap-3 border-b border-zinc-900 pb-4">
        <div className="p-2 bg-telemetry-orange/10 rounded-lg text-telemetry-orange">
          <BarChart3 className="w-5 h-5" />
        </div>
        <div>
          <h2 className="text-lg font-bold text-white uppercase tracking-wider">Métricas e Incidencias</h2>
          <p className="text-[10px] text-zinc-500 font-mono">SYSTEM_PERFORMANCE_ANALYTICS</p>
        </div>
      </div>

      {/* Modern Bento Grid KPI Metrics Ribbon */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-5">
        {/* KPI 1 */}
        <div className="glass-panel rounded-2xl p-5 flex items-center justify-between hover-trigger transition-all duration-300">
          <div>
            <span className="text-[9px] font-mono text-zinc-500 uppercase tracking-widest block">Inspecciones Totales</span>
            <span className="text-3xl font-mono font-bold text-white mt-1 block">{history.length}</span>
            <span className="text-[9px] font-mono text-zinc-600 block mt-1">TERMOGRAFÍAS EN ARCHIVO</span>
          </div>
          <div className="p-3 bg-telemetry-cyan/10 rounded-xl text-telemetry-cyan border border-telemetry-cyan/10">
            <FileText className="w-5 h-5" />
          </div>
        </div>
        {/* KPI 2 */}
        <div className="glass-panel rounded-2xl p-5 flex items-center justify-between hover-trigger transition-all duration-300">
          <div>
            <span className="text-[9px] font-mono text-zinc-500 uppercase tracking-widest block">Total de Anomalías</span>
            <span className="text-3xl font-mono font-bold text-telemetry-orange mt-1 block">{statsData.total}</span>
            <span className="text-[9px] font-mono text-zinc-600 block mt-1">HALLAZGOS CONFIRMADOS</span>
          </div>
          <div className="p-3 bg-telemetry-orange/10 rounded-xl text-telemetry-orange border border-telemetry-orange/10">
            <AlertTriangle className="w-5 h-5" />
          </div>
        </div>
        {/* KPI 3 */}
        <div className="glass-panel rounded-2xl p-5 flex items-center justify-between hover-trigger transition-all duration-300">
          <div>
            <span className="text-[9px] font-mono text-zinc-500 uppercase tracking-widest block">Eficiencia Estimada</span>
            <span className={`text-3xl font-mono font-bold mt-1 block ${statsData.solarEfficiency > 90 ? 'text-telemetry-green' : 'text-yellow-500'}`}>
              {statsData.solarEfficiency.toFixed(1)}%
            </span>
            <span className="text-[9px] font-mono text-zinc-600 block mt-1">RENDIMIENTO DEL PARQUE</span>
          </div>
          <div className="p-3 bg-telemetry-green/10 rounded-xl text-telemetry-green border border-telemetry-green/10">
            <Sun className="w-5 h-5" />
          </div>
        </div>
        {/* KPI 4 */}
        <div className="glass-panel rounded-2xl p-5 flex items-center justify-between hover-trigger transition-all duration-300">
          <div>
            <span className="text-[9px] font-mono text-zinc-500 uppercase tracking-widest block">Confianza IA</span>
            <span className="text-3xl font-mono font-bold text-telemetry-indigo mt-1 block">
              {statsData.avgConfidence > 0 ? `${statsData.avgConfidence.toFixed(1)}%` : '0.0%'}
            </span>
            <span className="text-[9px] font-mono text-zinc-600 block mt-1">CONFIA_MED_DETECCIONES</span>
          </div>
          <div className="p-3 bg-telemetry-indigo/10 rounded-xl text-telemetry-indigo border border-telemetry-indigo/10">
            <Cpu className="w-5 h-5" />
          </div>
        </div>
      </div>

      {/* Bento Grid Charts Layout */}
      <div className="grid lg:grid-cols-12 gap-6">
        {/* Distribution Bar Chart */}
        <section className="glass-panel rounded-2xl p-6 lg:col-span-7 h-[420px] flex flex-col">
          <h3 className="text-xs font-mono uppercase tracking-widest text-zinc-400 mb-6 flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-telemetry-orange" /> Frecuencia de Incidencias
          </h3>
          {statsData.barData.length > 0 ? (
            <div className="flex-1 w-full text-[10px] font-mono">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={statsData.barData} margin={{ top: 10, right: 10, left: -25, bottom: 5 }}>
                  <defs>
                    <linearGradient id="barGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#f97316" stopOpacity="0.8" />
                      <stop offset="100%" stopColor="#ef4444" stopOpacity="0.3" />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" vertical={false} />
                  <XAxis 
                    dataKey="name" 
                    stroke="#52525b" 
                    fontSize={9} 
                    tickLine={false}
                    tickFormatter={(val) => val.split(' ').map((s: string) => s[0]).join('')} // Initials
                  />
                  <YAxis stroke="#52525b" fontSize={9} axisLine={false} tickLine={false} />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: 'rgba(10, 10, 12, 0.9)', 
                      border: '1px solid rgba(255,255,255,0.08)', 
                      borderRadius: '12px',
                      backdropFilter: 'blur(8px)',
                      color: '#fff',
                      fontFamily: 'JetBrains Mono'
                    }}
                    cursor={{ fill: 'rgba(255, 255, 255, 0.02)' }}
                    itemStyle={{ fontSize: '10px', color: '#f97316', fontWeight: 'bold' }}
                    labelStyle={{ fontSize: '9px', textTransform: 'uppercase', color: '#a1a1aa' }}
                  />
                  <Bar dataKey="count" radius={[6, 6, 0, 0]} fill="url(#barGrad)">
                    {statsData.barData.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center opacity-30 text-zinc-500">
              <BarChart3 className="w-10 h-10 mb-2" />
              <p className="text-xs font-mono uppercase tracking-wider">Cero datos de distribución</p>
            </div>
          )}
        </section>

        {/* Mix percentage Pie Chart */}
        <section className="glass-panel rounded-2xl p-6 lg:col-span-5 h-[420px] flex flex-col">
          <h3 className="text-xs font-mono uppercase tracking-widest text-zinc-400 mb-6 flex items-center gap-2">
            <LayoutDashboard className="w-4 h-4 text-telemetry-cyan" /> Proporción de Fallas (%)
          </h3>
          {statsData.barData.length > 0 ? (
            <div className="flex-1 w-full text-[9px] font-mono flex flex-col justify-between">
              <div className="flex-1 relative w-full h-[220px]">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={statsData.barData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ percent }) => `${(percent * 100).toFixed(0)}%`}
                      outerRadius={80}
                      innerRadius={45}
                      paddingAngle={3}
                      fill="#8884d8"
                      dataKey="count"
                    >
                      {statsData.barData.map((_, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} stroke="rgba(0,0,0,0)" />
                      ))}
                    </Pie>
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'rgba(10, 10, 12, 0.9)', 
                        border: '1px solid rgba(255,255,255,0.08)', 
                        borderRadius: '12px',
                        backdropFilter: 'blur(8px)',
                        fontFamily: 'JetBrains Mono'
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="grid grid-cols-2 gap-2 mt-4 px-2 overflow-y-auto max-h-[80px]">
                {statsData.barData.map((d, index) => (
                  <div key={index} className="flex items-center gap-2 text-[8px] truncate">
                    <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ backgroundColor: COLORS[index % COLORS.length] }} />
                    <span className="text-zinc-500 uppercase truncate flex-1">{d.name}</span>
                    <span className="text-zinc-300 font-bold">{d.count} ({d.percentage}%)</span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center opacity-30 text-zinc-500">
              <LayoutDashboard className="w-10 h-10 mb-2" />
              <p className="text-xs font-mono uppercase tracking-wider">Cero datos de proporción</p>
            </div>
          )}
        </section>
      </div>

      <div className="glass-panel rounded-2xl p-6 flex flex-col md:flex-row items-center gap-6 justify-between text-zinc-400">
        <div className="flex items-center gap-4">
          <div className="p-3 bg-telemetry-orange/5 rounded-xl border border-telemetry-orange/10 text-telemetry-orange flex-shrink-0">
            <Sun className="w-8 h-8 animate-spin" style={{ animationDuration: '40s' }} />
          </div>
          <div>
            <h4 className="text-sm font-bold text-white uppercase tracking-wider font-sans">Reporte General del Parque Fotovoltaico</h4>
            <p className="text-[10px] text-zinc-500 font-mono mt-0.5 leading-relaxed max-w-2xl">
              Inspecciones procesadas por el motor inteligente de SolarGuard. Los datos presentados representan la salud térmica acumulada del array de paneles y sirven para la planificación de mantenimiento preventivo.
            </p>
          </div>
        </div>
        <div className="bg-zinc-950/60 p-4 border border-zinc-900 rounded-xl font-mono text-center min-w-[150px] flex-shrink-0">
          <span className="text-[9px] text-zinc-500 uppercase tracking-widest block">Índice de Incidencia</span>
          <span className="text-2xl font-bold text-telemetry-orange block mt-1">{(statsData.total / Math.max(1, history.length)).toFixed(2)}</span>
          <span className="text-[8px] text-zinc-600 block mt-0.5">FALLAS / INSPECCIÓN</span>
        </div>
      </div>
    </motion.div>
  );

  return (
    <div id="app-root" className="min-h-screen bg-[#050506] text-[#e4e4e7] font-sans selection:bg-telemetry-orange selection:text-white hud-grid-overlay overflow-x-hidden">
      {/* Sidebar Navigation */}
      <nav 
        id="navbar" 
        className="fixed left-0 top-0 bottom-0 w-20 border-r border-zinc-900 bg-[#070709]/85 backdrop-blur-md z-40 flex flex-col items-center py-6 gap-10"
      >
        <div className="p-2.5 bg-telemetry-orange rounded-xl shadow-[0_0_20px_rgba(249,115,22,0.3)] animate-pulse active-press">
          <Sun className="w-6 h-6 text-white" />
        </div>
        
        <div className="flex flex-col gap-3 relative">
          {/* Diagnostic Button */}
          <div className="relative">
            <button 
              onClick={() => setCurrentView('upload')}
              onMouseEnter={() => showTooltip('diag')}
              onMouseLeave={hideTooltip}
              className={`p-3.5 rounded-xl transition-all duration-200 active-press ${currentView === 'upload' ? 'bg-telemetry-orange text-white shadow-[0_0_15px_rgba(249,115,22,0.35)]' : 'text-zinc-500 hover:bg-zinc-900 hover:text-zinc-200'}`}
            >
              <ShieldAlert className="w-5.5 h-5.5" />
            </button>
            <AnimatePresence>
              {hoveredTooltip === 'diag' && (
                <motion.div 
                  initial={{ opacity: 0, scale: 0.95, x: 10 }}
                  animate={{ opacity: 1, scale: 1, x: 0 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  transition={{ duration: anyTooltipOpen ? 0 : 0.15, ease: 'easeOut' }}
                  className="absolute left-16 top-3 bg-zinc-950 border border-zinc-800 text-zinc-200 text-[10px] font-mono uppercase tracking-widest px-3 py-1.5 rounded-lg whitespace-nowrap shadow-xl pointer-events-none z-50"
                >
                  Diagnóstico IA
                </motion.div>
              )}
            </AnimatePresence>
          </div>
          
          {/* History Button */}
          <div className="relative">
            <button 
              onClick={() => setCurrentView('history')}
              onMouseEnter={() => showTooltip('hist')}
              onMouseLeave={hideTooltip}
              className={`p-3.5 rounded-xl transition-all duration-200 active-press ${currentView === 'history' ? 'bg-telemetry-orange text-white shadow-[0_0_15px_rgba(249,115,22,0.35)]' : 'text-zinc-500 hover:bg-zinc-900 hover:text-zinc-200'}`}
            >
              <HistoryIcon className="w-5.5 h-5.5" />
            </button>
            <AnimatePresence>
              {hoveredTooltip === 'hist' && (
                <motion.div 
                  initial={{ opacity: 0, scale: 0.95, x: 10 }}
                  animate={{ opacity: 1, scale: 1, x: 0 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  transition={{ duration: anyTooltipOpen ? 0 : 0.15, ease: 'easeOut' }}
                  className="absolute left-16 top-3 bg-zinc-950 border border-zinc-800 text-zinc-200 text-[10px] font-mono uppercase tracking-widest px-3 py-1.5 rounded-lg whitespace-nowrap shadow-xl pointer-events-none z-50"
                >
                  Historial
                </motion.div>
              )}
            </AnimatePresence>
          </div>
          
          {/* Stats Button */}
          <div className="relative">
            <button 
              onClick={() => setCurrentView('stats')}
              onMouseEnter={() => showTooltip('stats')}
              onMouseLeave={hideTooltip}
              className={`p-3.5 rounded-xl transition-all duration-200 active-press ${currentView === 'stats' ? 'bg-telemetry-orange text-white shadow-[0_0_15px_rgba(249,115,22,0.35)]' : 'text-zinc-500 hover:bg-zinc-900 hover:text-zinc-200'}`}
            >
              <BarChart3 className="w-5.5 h-5.5" />
            </button>
            <AnimatePresence>
              {hoveredTooltip === 'stats' && (
                <motion.div 
                  initial={{ opacity: 0, scale: 0.95, x: 10 }}
                  animate={{ opacity: 1, scale: 1, x: 0 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  transition={{ duration: anyTooltipOpen ? 0 : 0.15, ease: 'easeOut' }}
                  className="absolute left-16 top-3 bg-zinc-950 border border-zinc-800 text-zinc-200 text-[10px] font-mono uppercase tracking-widest px-3 py-1.5 rounded-lg whitespace-nowrap shadow-xl pointer-events-none z-50"
                >
                  Estadísticas
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        <div className="mt-auto items-center flex flex-col gap-5">
           <div className="w-2 h-2 rounded-full bg-telemetry-green animate-pulse" />
           <p className="text-[7.5px] font-mono text-zinc-600 tracking-tighter -rotate-90 origin-center whitespace-nowrap">SG_NODE_V10.2_ACTIVE</p>
        </div>
      </nav>

      {/* Main Content Pane */}
      <div className="pl-20 min-h-screen flex flex-col">
        {/* Header HUD */}
        <header id="header" className="border-b border-zinc-900 bg-[#070709]/80 backdrop-blur-md sticky top-0 z-30">
          <div className="max-w-7xl mx-auto px-8 h-20 flex items-center justify-between">
            <div>
              <h1 className="text-base font-bold tracking-wider text-white uppercase font-sans">
                {currentView === 'upload' && 'Consola de Diagnóstico'}
                {currentView === 'history' && 'Memoria de Inspección'}
                {currentView === 'stats' && 'Panel Métrico de Anomalías'}
              </h1>
              <p className="text-[9px] text-zinc-500 uppercase tracking-[0.25em] font-mono mt-0.5 leading-none">
                SolarGuard Intelligence telemetry · Node 1
              </p>
            </div>
            
            <div className="flex items-center gap-4 text-[9px] font-mono text-zinc-500">
               <div className="hidden sm:flex items-center gap-2">
                 <span className="w-1.5 h-1.5 rounded-full bg-telemetry-green" />
                 <span>API_STATUS: <span className="text-telemetry-green font-bold">CONECTADO</span></span>
               </div>
               <span className="h-3.5 w-[1px] bg-zinc-800 hidden sm:inline" />
               <span className="bg-zinc-950 border border-zinc-900 px-2 py-0.5 rounded text-zinc-400">
                 ENGINE: YOLOv9-S
               </span>
            </div>
          </div>
        </header>

        {/* Main Section */}
        <main className="max-w-7xl w-full mx-auto px-8 py-8 flex-1">
          <AnimatePresence mode="wait">
            {currentView === 'upload' && (
              <motion.div 
                key="upload-view" 
                initial={{ opacity: 0, y: 5 }} 
                animate={{ opacity: 1, y: 0 }} 
                exit={{ opacity: 0, y: -5 }} 
                transition={{ duration: 0.15, ease: 'easeOut' }}
              >
                <UploadView />
              </motion.div>
            )}
            {currentView === 'history' && (
              <motion.div 
                key="history-view" 
                initial={{ opacity: 0, y: 5 }} 
                animate={{ opacity: 1, y: 0 }} 
                exit={{ opacity: 0, y: -5 }} 
                transition={{ duration: 0.15, ease: 'easeOut' }}
              >
                <HistoryView />
              </motion.div>
            )}
            {currentView === 'stats' && (
              <motion.div 
                key="stats-view" 
                initial={{ opacity: 0, y: 5 }} 
                animate={{ opacity: 1, y: 0 }} 
                exit={{ opacity: 0, y: -5 }} 
                transition={{ duration: 0.15, ease: 'easeOut' }}
              >
                <StatsView />
              </motion.div>
            )}
          </AnimatePresence>
        </main>
      </div>
      
      {/* Decorative Radial Background Gradient Glows */}
      <div className="fixed inset-0 pointer-events-none z-[-1] overflow-hidden opacity-[0.03]">
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-[900px] h-[900px] bg-telemetry-orange rounded-full blur-[160px] animate-ambient-glow" />
      </div>
    </div>
  );
}



