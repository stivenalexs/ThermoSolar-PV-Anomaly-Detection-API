/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useRef, useEffect, useMemo } from 'react';
import axios from 'axios';
import { 
  Upload, Sun, ShieldAlert, CheckCircle2, History as HistoryIcon, 
  AlertTriangle, Info, LayoutDashboard, BarChart3, Clock, Calendar, 
  Trash2, FileText, ChevronRight
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
  imageUrl?: string; // We'll store dataURL for simple local history if needed
}

type View = 'upload' | 'history' | 'stats';

// --- Constants ---

const COLORS = ['#f97316', '#3b82f6', '#10b981', '#ef4444', '#8b5cf6', '#eab308', '#ec4899', '#06b6d4'];

const CLASS_NAMES_LIST = [
  "Single Hotspot", "Multi Hotspots", "Single Diode", "Multi Diode",
  "Single Bypassed Substring", "Multi Bypassed Substring", "String Open Circuit", "String Reversed Polarity"
];

// --- Helper Functions ---

const formatDate = (date: Date) => {
  const days = ['Domingo', 'Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado'];
  return {
    date: date.toLocaleDateString('es-ES'),
    time: date.toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit' }),
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
  const fileInputRef = useRef<HTMLInputElement>(null);

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

  const handleUpload = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post<ApiResponse>('/predict', formData, {
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
      setError(err.response?.data?.detail || 'Error al conectar con la API. Asegúrate de que el servidor Python esté corriendo.');
    } finally {
      setLoading(false);
    }
  };

  const clearHistory = () => {
    if (window.confirm('¿Estás seguro de que quieres borrar todo el historial?')) {
      setHistory([]);
    }
  };

  // --- Computed Stats ---
  const statsData = useMemo(() => {
    const counts: Record<string, number> = {};
    CLASS_NAMES_LIST.forEach(name => (counts[name] = 0));
    
    history.forEach(item => {
      item.detections.forEach(d => {
        counts[d.class_name] = (counts[d.class_name] || 0) + 1;
      });
    });

    const total = Object.values(counts).reduce((acc, curr) => acc + curr, 0);

    const barData = Object.entries(counts).map(([name, count]) => ({
      name,
      count,
      percentage: total > 0 ? ((count / total) * 100).toFixed(1) : 0
    })).filter(d => d.count > 0);

    return { barData, total };
  }, [history]);

  // --- Views ---

  const UploadView = () => (
    <div className="grid lg:grid-cols-12 gap-8">
      {/* Control Panel */}
      <div className="lg:col-span-4 space-y-6">
        <section id="upload-section" className="bg-[#141414] border border-gray-800 rounded-2xl p-6 shadow-xl">
          <div className="flex items-center gap-2 mb-6">
            <Upload className="w-5 h-5 text-orange-500" />
            <h2 className="font-semibold">Carga de Datos</h2>
          </div>

          <div 
            id="drop-zone"
            onClick={() => fileInputRef.current?.click()}
            className={`
              border-2 border-dashed rounded-xl p-8 transition-all cursor-pointer group
              ${selectedFile ? 'border-orange-500/50 bg-orange-500/5' : 'border-gray-800 hover:border-gray-700 bg-gray-900/50'}
            `}
          >
            <input 
              type="file" 
              ref={fileInputRef}
              onChange={handleFileSelect}
              className="hidden" 
              accept="image/*"
            />
            <div className="flex flex-col items-center text-center">
              <div className={`p-4 rounded-full mb-4 transition-colors ${selectedFile ? 'bg-orange-500 text-white' : 'bg-gray-800 text-gray-500 group-hover:text-gray-300'}`}>
                <Upload className="w-8 h-8" />
              </div>
              <p className="font-medium mb-1 text-sm">
                {selectedFile ? selectedFile.name : 'Seleccionar Imagen Térmica'}
              </p>
              <p className="text-[10px] text-gray-500 uppercase tracking-tighter">JPG, PNG o GIF (max 10MB)</p>
            </div>
          </div>

          <button
            id="analyze-button"
            disabled={!selectedFile || loading}
            onClick={handleUpload}
            className={`
              w-full mt-6 py-4 rounded-xl font-bold flex items-center justify-center gap-3 transition-all
              ${!selectedFile || loading 
                ? 'bg-gray-800 text-gray-500 cursor-not-allowed' 
                : 'bg-orange-500 hover:bg-orange-600 text-white shadow-[0_0_20px_rgba(249,115,22,0.3)]'}
            `}
          >
            {loading ? (
              <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            ) : (
              <ShieldAlert className="w-5 h-5" />
            )}
            {loading ? 'Analizando Red Neuronal...' : 'Ejecutar Diagnóstico'}
          </button>
        </section>

        {/* System Info */}
        <section id="info-section" className="bg-[#141414] border border-gray-800 rounded-2xl p-6">
          <div className="flex items-center gap-2 mb-4">
            <Info className="w-5 h-5 text-gray-400" />
            <h2 className="font-semibold text-sm uppercase tracking-wide">Motor de Análisis</h2>
          </div>
          <div className="space-y-3 font-mono text-[11px]">
            <div className="flex justify-between py-2 border-b border-gray-800/50 text-gray-500">
              <span>Arquitectura</span>
              <span className="text-gray-300">YOLOv9-Small</span>
            </div>
            <div className="flex justify-between py-2 border-b border-gray-800/50 text-gray-500">
              <span>Resolución</span>
              <span className="text-gray-300">640x640px</span>
            </div>
            <div className="flex justify-between py-2 border-b border-gray-800/50 text-gray-500">
              <span>Categorías</span>
              <span className="text-gray-300">8 Anomalías</span>
            </div>
          </div>
        </section>
      </div>

      {/* Results Panel */}
      <div className="lg:col-span-8 flex flex-col gap-6">
        <section id="view-section" className="bg-[#141414] border border-gray-800 rounded-2xl overflow-hidden flex-1 min-h-[400px] relative">
          <div className="bg-[#0f0f0f] border-b border-gray-800 px-6 py-4 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-red-500/50" />
              <div className="w-3 h-3 rounded-full bg-yellow-500/50" />
              <div className="w-3 h-3 rounded-full bg-green-500/50" />
              <span className="ml-3 text-xs font-mono text-gray-500 uppercase tracking-widest">Viewport de Diagnóstico</span>
            </div>
            {results && (
              <div className="bg-orange-500/10 text-orange-500 text-[10px] px-2 py-0.5 rounded font-mono border border-orange-500/20">
                PROCESO_COMPLETO
              </div>
            )}
          </div>

          <div className="p-8 flex items-center justify-center min-h-[340px]">
            {!previewUrl ? (
              <div className="text-center opacity-30 px-12">
                <Sun className="w-24 h-24 mx-auto mb-4" />
                <p className="text-lg font-light tracking-wide italic">Modo Standby. Esperando datos de entrada...</p>
              </div>
            ) : (
              <div className="relative max-w-full rounded-lg overflow-hidden group">
                <img src={previewUrl} className="max-h-[60vh] object-contain shadow-2xl transition-transform duration-700 group-hover:scale-[1.01]" alt="Preview" />
                
                {/* Bounding Boxes Overlay */}
                {results && results.detections.map((d, index) => (
                  <div 
                    key={index}
                    className="absolute border-2 border-orange-500 pointer-events-none transition-opacity"
                    style={{
                      left: `${(d.bbox.x_min / 640) * 100}%`,
                      top: `${(d.bbox.y_min / 640) * 100}%`,
                      width: `${(d.bbox.width / 640) * 100}%`,
                      height: `${(d.bbox.height / 640) * 100}%`,
                    }}
                  >
                    <span className="absolute -top-6 left-0 bg-orange-500 text-white text-[9px] px-1 font-mono uppercase font-bold whitespace-nowrap shadow-lg">
                      {d.class_name} ({(d.confidence * 100).toFixed(1)}%)
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </section>

        {/* Results Summary */}
        <section id="analysis-results" className="bg-[#141414] border border-gray-800 rounded-2xl p-6 min-h-[160px]">
          <AnimatePresence mode="wait">
            {results ? (
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                key="results"
                className="space-y-4"
              >
                <div className="flex items-center justify-between font-mono">
                  <h3 className="text-sm uppercase tracking-widest text-gray-500">Hallazgos de IA</h3>
                  <span className="text-xs text-orange-500">{results.detections.length} Anomalías Detectadas</span>
                </div>
                
                <div className="grid sm:grid-cols-2 gap-3 pb-4">
                  {results.detections.length > 0 ? (
                    results.detections.map((d, i) => (
                      <div key={i} className="bg-gray-900 border border-gray-800 rounded-xl p-3 flex items-center gap-4 hover:border-gray-700 transition-colors cursor-default group">
                        <div className="bg-orange-500/10 p-2 rounded-lg text-orange-500 group-hover:bg-orange-500 group-hover:text-white transition-colors">
                          <AlertTriangle className="w-5 h-5" />
                        </div>
                        <div>
                          <p className="text-sm font-semibold">{d.class_name}</p>
                          <p className="text-[10px] text-gray-500 font-mono tracking-tighter uppercase">Confianza: {(d.confidence * 100).toFixed(2)}% | ID: {d.class_id}</p>
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="col-span-2 py-8 flex flex-col items-center justify-center opacity-40 grayscale">
                      <CheckCircle2 className="w-12 h-12 mb-3 text-green-500" />
                      <p className="text-sm font-medium italic">Sistema estable. No se detectaron fallas térmicas.</p>
                    </div>
                  )}
                </div>
              </motion.div>
            ) : error ? (
              <motion.div 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                key="error"
                className="p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-500 flex items-start gap-3"
              >
                <AlertTriangle className="w-5 h-5 mt-0.5 flex-shrink-0" />
                <div className="space-y-1">
                  <p className="text-sm font-bold uppercase tracking-widest">Error de Conexión</p>
                  <p className="text-xs opacity-80 leading-relaxed font-mono">{error}</p>
                </div>
              </motion.div>
            ) : (
              <div className="flex flex-col items-center justify-center h-full py-8 opacity-20">
                <Clock className="w-8 h-8 mb-2" />
                <p className="text-xs uppercase tracking-[0.2em]">Esperando flujo de telemetría</p>
              </div>
            )}
          </AnimatePresence>
        </section>
      </div>
    </div>
  );

  const HistoryView = () => (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <HistoryIcon className="w-6 h-6 text-orange-500" />
          <h2 className="text-2xl font-bold">Historial de Detecciones</h2>
        </div>
        <button 
          onClick={clearHistory}
          disabled={history.length === 0}
          className="flex items-center gap-2 px-4 py-2 bg-red-500/10 text-red-500 rounded-lg hover:bg-red-500 hover:text-white transition-all disabled:opacity-30 disabled:cursor-not-allowed text-sm font-medium"
        >
          <Trash2 className="w-4 h-4" /> Borrar Todo
        </button>
      </div>

      <div className="bg-[#141414] border border-gray-800 rounded-2xl overflow-hidden">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-gray-800 bg-gray-900/50 text-gray-500 text-xs uppercase font-mono tracking-widest">
              <th className="px-6 py-4">Archivo</th>
              <th className="px-6 py-4">Fecha / Día</th>
              <th className="px-6 py-4">Hora</th>
              <th className="px-6 py-4">Hallazgos</th>
              <th className="px-6 py-4 text-right">Detalle</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800">
            {history.length > 0 ? (
              history.map((item) => (
                <tr key={item.id} className="hover:bg-gray-800/30 transition-colors group">
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-3">
                      <FileText className="w-4 h-4 text-gray-500" />
                      <span className="text-sm font-medium text-gray-300">{item.filename}</span>
                    </div>
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-400">
                    <div className="flex flex-col">
                      <span>{item.date}</span>
                      <span className="text-[10px] text-gray-600 uppercase font-mono">{item.day}</span>
                    </div>
                  </td>
                  <td className="px-6 py-4 text-sm font-mono text-gray-500">
                    {item.time}
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-2">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${item.detections.length > 0 ? 'bg-orange-500/10 text-orange-500' : 'bg-green-500/10 text-green-500'}`}>
                        {item.detections.length} ANOMALÍAS
                      </span>
                    </div>
                  </td>
                  <td className="px-6 py-4 text-right">
                    <button className="text-gray-600 hover:text-white transition-colors">
                      <ChevronRight className="w-5 h-5 ml-auto" />
                    </button>
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={5} className="px-6 py-20 text-center text-gray-600">
                  <div className="flex flex-col items-center gap-2 opacity-50">
                    <Clock className="w-12 h-12" />
                    <p className="text-sm font-light">No hay registros de análisis previos.</p>
                  </div>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </motion.div>
  );

  const StatsView = () => (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-8">
      <div className="flex items-center gap-3">
        <BarChart3 className="w-6 h-6 text-orange-500" />
        <h2 className="text-2xl font-bold">Estadísticas de Campo</h2>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        {/* Bar Chart */}
        <section className="bg-[#141414] border border-gray-800 rounded-2xl p-6 h-[400px] flex flex-col">
          <h3 className="text-sm font-mono uppercase tracking-widest text-gray-500 mb-6 flex items-center gap-2">
            <BarChart3 className="w-4 h-4" /> Distribución de Incidencias
          </h3>
          <div className="flex-1 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={statsData.barData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#222" vertical={false} />
                <XAxis 
                  dataKey="name" 
                  stroke="#555" 
                  fontSize={10} 
                  tickFormatter={(val) => val.split(' ').map((s: string) => s[0]).join('')} // Initials for small screens
                />
                <YAxis stroke="#555" fontSize={10} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#141414', border: '1px solid #333', borderRadius: '8px' }}
                  itemStyle={{ fontSize: '11px', fontWeight: 'bold' }}
                />
                <Bar dataKey="count" radius={[4, 4, 0, 0]} fill="#f97316">
                  {statsData.barData.map((_, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>

        {/* Pie/Pastel Chart */}
        <section className="bg-[#141414] border border-gray-800 rounded-2xl p-6 h-[400px] flex flex-col">
          <h3 className="text-sm font-mono uppercase tracking-widest text-gray-500 mb-6 flex items-center gap-2">
            <LayoutDashboard className="w-4 h-4" /> Mix de Anomalías (%)
          </h3>
          <div className="flex-1 w-full relative">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={statsData.barData}
                  cx="50%"
                  cy="45%"
                  labelLine={false}
                  label={({ name, percent }) => `${(percent * 100).toFixed(0)}%`}
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="count"
                >
                  {statsData.barData.map((_, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} stroke="rgba(0,0,0,0)" />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ backgroundColor: '#141414', border: '1px solid #333', borderRadius: '8px' }}
                />
                <Legend layout="horizontal" verticalAlign="bottom" align="center" wrapperStyle={{ fontSize: '10px' }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>

      <div className="bg-[#141414] border border-gray-800 rounded-2xl p-8 text-center">
        <Sun className="w-12 h-12 text-orange-500 mx-auto mb-4 opacity-50" />
        <h4 className="text-xl font-bold mb-2">Total Analizado: {history.length} Imágenes</h4>
        <p className="text-gray-500 text-sm max-w-md mx-auto">
          Resumen acumulado del rendimiento y fallas detectadas en el parque fotovoltaico. 
          Un total de <span className="text-orange-500 font-bold">{statsData.total} incidencias</span> térmicas han sido procesadas por el motor SolarGuard.
        </p>
      </div>
    </motion.div>
  );

  return (
    <div id="app-root" className="min-h-screen bg-[#0a0a0a] text-gray-100 font-sans selection:bg-orange-500 selection:text-white">
      {/* Sidebar Navigation */}
      <nav id="navbar" className="fixed left-0 top-0 bottom-0 w-20 border-r border-gray-800 bg-[#0f0f0f] z-50 flex flex-col items-center py-8 gap-10">
        <div className="p-3 bg-orange-500 rounded-2xl shadow-[0_0_20px_rgba(249,115,22,0.4)]">
          <Sun className="w-8 h-8 text-white" />
        </div>
        
        <div className="flex flex-col gap-4">
          <button 
            onClick={() => setCurrentView('upload')}
            className={`p-4 rounded-2xl transition-all ${currentView === 'upload' ? 'bg-orange-500 text-white' : 'text-gray-500 hover:bg-gray-800 hover:text-white'}`}
            title="Diagnóstico"
          >
            <ShieldAlert className="w-6 h-6" />
          </button>
          
          <button 
            onClick={() => setCurrentView('history')}
            className={`p-4 rounded-2xl transition-all ${currentView === 'history' ? 'bg-orange-500 text-white' : 'text-gray-500 hover:bg-gray-800 hover:text-white'}`}
            title="Historial"
          >
            <Clock className="w-6 h-6" />
          </button>
          
          <button 
            onClick={() => setCurrentView('stats')}
            className={`p-4 rounded-2xl transition-all ${currentView === 'stats' ? 'bg-orange-500 text-white' : 'text-gray-500 hover:bg-gray-800 hover:text-white'}`}
            title="Estadísticas"
          >
            <BarChart3 className="w-6 h-6" />
          </button>
        </div>

        <div className="mt-auto items-center flex flex-col gap-4">
           <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
           <p className="text-[8px] font-mono text-gray-700 tracking-tighter -rotate-90 origin-center whitespace-nowrap">REMOTE_NODE_v1.0</p>
        </div>
      </nav>

      <div className="pl-20">
        {/* Header */}
        <header id="header" className="border-b border-gray-800 bg-[#0f0f0f]/80 backdrop-blur-md sticky top-0 z-40">
          <div className="max-w-7xl mx-auto px-10 h-20 flex items-center justify-between">
            <div>
              <h1 className="text-xl font-bold tracking-tight">
                {currentView === 'upload' && 'Sistema de Diagnóstico'}
                {currentView === 'history' && 'Histórico de Inspección'}
                {currentView === 'stats' && 'Métricas de Incidencia'}
              </h1>
              <p className="text-[10px] text-gray-500 uppercase tracking-[0.3em] font-mono leading-none">
                SolarGuard Intelligence Engine v9.0.2
              </p>
            </div>
            
            <div className="flex items-center gap-4 text-xs font-mono text-gray-600">
               <span className="hidden sm:inline">AUTOMA_STATUS: <span className="text-green-500">READY</span></span>
               <span className="h-4 w-[1px] bg-gray-800 hidden sm:inline" />
               <span>USER: {new Date().getHours() > 18 ? 'INS_NIGHT' : 'INS_DAY'}</span>
            </div>
          </div>
        </header>

        <main className="max-w-7xl mx-auto px-10 py-10">
          <AnimatePresence mode="wait">
            {currentView === 'upload' && <motion.div key="u" initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 10 }} transition={{ duration: 0.2 }}><UploadView /></motion.div>}
            {currentView === 'history' && <motion.div key="h" initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 10 }} transition={{ duration: 0.2 }}><HistoryView /></motion.div>}
            {currentView === 'stats' && <motion.div key="s" initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 10 }} transition={{ duration: 0.2 }}><StatsView /></motion.div>}
          </AnimatePresence>
        </main>
      </div>
      
      {/* Visual background element */}
      <div className="fixed inset-0 pointer-events-none z-[-1] overflow-hidden opacity-10">
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-[800px] h-[800px] bg-orange-600/30 rounded-full blur-[150px]" />
      </div>
    </div>
  );
}


