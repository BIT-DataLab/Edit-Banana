
import React, { useState, useEffect, useReducer, useRef } from 'react';
import axios from 'axios';
import DrawioEditor from './DrawioEditor';
import { Upload, FileImage, Loader2, CheckCircle, AlertCircle, Eye, EyeOff, Layout, Plus, Image as ImageIcon, ChevronLeft, ChevronRight } from 'lucide-react';
import './App.css'

const API_BASE = "http://localhost:8001";

// --- Reducer for Managing Tasks ---
// Actions: ADD_TASKS, UPDATE_TASK, SELECT_TASK
const tasksReducer = (state, action) => {
  switch (action.type) {
    case 'ADD_TASKS':
      const newTasks = action.payload.map(file => ({
        id: Math.random().toString(36).substr(2, 9),
        file,
        name: file.name,
        previewUrl: URL.createObjectURL(file), // Create local URL for preview
        status: 'idle', // idle, uploading, processing, completed, error
        progress: 0,
        statusMessage: 'Ready',
        backendTaskId: null,
        resultXml: null,
        error: null
      }));
      return {
        ...state,
        tasks: [...state.tasks, ...newTasks],
        activeTaskId: state.activeTaskId || newTasks[0].id // Auto select first if none selected
      };
    case 'UPDATE_TASK':
      return {
        ...state,
        tasks: state.tasks.map(t => 
          t.id === action.payload.id ? { ...t, ...action.payload.updates } : t
        )
      };
    case 'SELECT_TASK':
      return {
        ...state,
        activeTaskId: action.payload
      };
    default:
      return state;
  }
};

function App() {
  const [state, dispatch] = useReducer(tasksReducer, {
    tasks: [],
    activeTaskId: null
  });

  const [backendStatus, setBackendStatus] = useState("checking");
  const [isForceMode, setIsForceMode] = useState(false); // Force re-process state
  
  // Layout State
  const [sidebarWidth, setSidebarWidth] = useState(250);
  const [showSidebar, setShowSidebar] = useState(true);
  
  const [imagePaneWidth, setImagePaneWidth] = useState(450); // px
  const [showOriginal, setShowOriginal] = useState(true);
  const [showEditor, setShowEditor] = useState(true);
  const [editorCommand, setEditorCommand] = useState(null);

  // Resize Refs
  const sidebarRef = useRef(null);
  const isResizingSidebar = useRef(false);
  
  const imagePaneRef = useRef(null);
  const isResizingImagePane = useRef(false);

  // Derived state
  const activeTask = state.tasks.find(t => t.id === state.activeTaskId);

  useEffect(() => {
    // Check backend status
    axios.get(`${API_BASE}/`)
      .then(() => setBackendStatus("Online"))
      .catch(err => setBackendStatus("Offline"));
  }, []);

  // --- Resize Logic ---
  useEffect(() => {
    const handleMouseMove = (e) => {
      // 1. Sidebar Resize
      if (isResizingSidebar.current) {
         // Constrain width
         const newWidth = Math.max(150, Math.min(e.clientX, 600));
         setSidebarWidth(newWidth);
      }

      // 2. Image Pane Resize
      if (isResizingImagePane.current) {
         // Calculate offset from left side of viewport
         // If sidebar is visible, offset is sidebarWidth. If not, 0.
         const offset = showSidebar ? sidebarWidth : 0;
         const relativeX = e.clientX - offset;
         
         const newWidth = Math.max(200, Math.min(relativeX, window.innerWidth - 300));
         setImagePaneWidth(newWidth);
      }
    };

    const handleMouseUp = () => {
      if (isResizingSidebar.current || isResizingImagePane.current) {
        isResizingSidebar.current = false;
        isResizingImagePane.current = false;
        document.body.style.cursor = 'default';
        document.body.style.userSelect = 'auto'; // Re-enable selection
      }
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [showSidebar, sidebarWidth]); // Dep on sidebar state for offset calculation

  const startResizingSidebar = (e) => {
     e.preventDefault();
     isResizingSidebar.current = true;
     document.body.style.cursor = 'col-resize';
     document.body.style.userSelect = 'none'; // Prevent selection while dragging
  };

  const startResizingImagePane = (e) => {
    e.preventDefault();
    isResizingImagePane.current = true;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  };


  // --- File Handling ---

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      dispatch({ type: 'ADD_TASKS', payload: Array.from(e.target.files) });
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      dispatch({ type: 'ADD_TASKS', payload: Array.from(e.dataTransfer.files) });
    }
  };

  // --- Processing Logic ---

  // Trigger processing for all 'idle' tasks
  useEffect(() => {
    state.tasks.forEach(task => {
      if (task.status === 'idle') {
        processTask(task);
      }
    });
  }, [state.tasks.length]); // Only check when list grows

  const processTask = async (task) => {
    // 1. Update status to uploading
    dispatch({ 
      type: 'UPDATE_TASK', 
      payload: { id: task.id, updates: { status: 'uploading', statusMessage: 'Uploading...', progress: 10 } } 
    });

    const formData = new FormData();
    formData.append("file", task.file);

    try {
      // Pass force parameter
      const uploadRes = await axios.post(`${API_BASE}/upload?force=${isForceMode}`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 60000 
      });

      const { task_id, cached } = uploadRes.data;
      
      dispatch({ 
        type: 'UPDATE_TASK', 
        payload: { 
          id: task.id, 
          updates: { 
            status: 'processing', 
            statusMessage: 'Processing...', 
            progress: 30, 
            backendTaskId: task_id 
          } 
        } 
      });

      // Start Polling
      if (cached) {
         // Instant complete if cached
         dispatch({ 
            type: 'UPDATE_TASK', 
            payload: { 
              id: task.id, 
              updates: { 
                 status: 'fetching_result', 
                 statusMessage: 'Loading from Cache...', 
                 progress: 99 
              } 
            } 
         });
         fetchResult(task.id, task_id);
      } else {
         pollStatus(task.id, task_id);
      }

    } catch (error) {
       dispatch({ 
        type: 'UPDATE_TASK', 
        payload: { 
          id: task.id, 
          updates: { 
            status: 'error', 
            statusMessage: "Upload failed", 
            error: error.message 
          } 
        } 
      });
    }
  };

  const pollStatus = (localId, backendId) => {
    const interval = setInterval(async () => {
      try {
        const res = await axios.get(`${API_BASE}/task/${backendId}`);
        const { status: taskStatus, error, progress: taskProgress } = res.data;

        let updates = {};
        let shouldStop = false;

        if (taskProgress) {
             updates.progress = 30 + (taskProgress * 60);
        }

        if (taskStatus === "completed") {
          shouldStop = true;
          updates.status = 'fetching_result';
          updates.statusMessage = 'Finalizing...';
          updates.progress = 95;
          // Trigger fetch result
          fetchResult(localId, backendId);
        } else if (taskStatus === "failed") {
          shouldStop = true;
          updates.status = 'error';
          updates.statusMessage = 'Failed: ' + error;
          updates.error = error;
        } else {
             updates.statusMessage = "The image is being processed...";
        }
        
        dispatch({ type: 'UPDATE_TASK', payload: { id: localId, updates }});

        if (shouldStop) clearInterval(interval);

      } catch (err) {
        console.error(err);
      }
    }, 2000);
  };

  const fetchResult = async (localId, backendId) => {
    try {
      const res = await axios.get(`${API_BASE}/files/${backendId}/xml`, {
        responseType: 'text' 
      });
      dispatch({ 
        type: 'UPDATE_TASK', 
        payload: { 
          id: localId, 
          updates: { 
            status: 'completed', 
            statusMessage: 'Done', 
            progress: 100,
            resultXml: res.data
          } 
        } 
      });
    } catch (error) {
      dispatch({ 
        type: 'UPDATE_TASK', 
        payload: { 
          id: localId, 
          updates: { status: 'error', statusMessage: 'XML Download Failed' } 
        } 
      });
    }
  };

  // --- Render Components ---

  const handleImageClick = async (e) => {
      // 仅在任务完成且有后端ID时处理
      if (!activeTask || !activeTask.backendTaskId || activeTask.status !== 'completed') return;

      const img = e.target;
      const rect = img.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      
      // Scale to natural size (backend expects original image coords)
      // 如果图片未加载完成 naturalWidth 可能为 0，需容错
      if (!img.naturalWidth) return;

      const scaleX = img.naturalWidth / rect.width;
      const scaleY = img.naturalHeight / rect.height;
      
      const realX = Math.round(x * scaleX);
      const realY = Math.round(y * scaleY);
      
      console.log(`Click at ${realX}, ${realY}`);
      
      try {
          // 显示加载状态或通知用户 (此处简单处理)
          document.body.style.cursor = 'wait';
          
          const res = await axios.post(`${API_BASE}/interactive/segment`, {
              task_id: activeTask.backendTaskId,
              x: realX,
              y: realY
          });
          
          document.body.style.cursor = 'default';

          if (res.data.status === 'success' && res.data.data) {
              const { base64, bbox } = res.data.data;
              // 构造合并用的XML
              const id = 'added_' + Date.now();
              // 注意：DrawIO中 shape=image 需要 image=data:image/png,base64str...
              // 我们后端返回的 base64 不带前缀，需要加上
              const style = `shape=image;verticalLabelPosition=bottom;verticalAlign=top;imageAspect=0;aspect=fixed;image=data:image/png;base64,${base64};`;
              const width = bbox[2] - bbox[0];
              const height = bbox[3] - bbox[1];
              const xPos = bbox[0];
              const yPos = bbox[1];
              
              // 使用 merge 动作将新单元格合并到现有图表中
              const xmlSnippet = `
              <mxGraphModel>
                <root>
                  <mxCell id="${id}" style="${style}" vertex="1" parent="1">
                    <mxGeometry x="${xPos}" y="${yPos}" width="${width}" height="${height}" as="geometry"/>
                  </mxCell>
                </root>
              </mxGraphModel>
              `;
              
              console.log("Segmented! Sending merge command...");
              setEditorCommand({ action: 'merge', xml: xmlSnippet });
          }
      } catch (err) {
          document.body.style.cursor = 'default';
          console.error("Segmentation failed", err);
          alert("点击分割失败，请重试。\nSegmentation failed: " + (err.response?.data?.detail || err.message));
      }
  };

  const renderSidebar = () => {
    if (!showSidebar) return null;
    return (
    <div className="sidebar" style={{width: sidebarWidth}} ref={sidebarRef}>
      <div className="sidebar-header">
         <span>Your Files</span>
         <button className="btn btn-icon btn-sm" onClick={() => document.getElementById('sidebarInput').click()}>
            <Plus size={18} />
         </button>
         <input 
            id="sidebarInput" 
            type="file" 
            accept="image/*" 
            multiple
            onChange={handleFileChange} 
            style={{display: 'none'}}
          />
      </div>
      <div className="task-list">
         {state.tasks.map(task => (
           <div 
             key={task.id} 
             className={`task-item ${state.activeTaskId === task.id ? 'active' : ''}`}
             onClick={() => dispatch({type: 'SELECT_TASK', payload: task.id})}
           >
              <img src={task.previewUrl} className="task-thumb" alt="" />
              <div className="task-info">
                 <div className="task-name">{task.name}</div>
                 <div className="task-status">
                    {task.status === 'completed' && <CheckCircle size={12} className="status-icon completed"/>}
                    {task.status === 'error' && <AlertCircle size={12} className="status-icon error"/>}
                    {task.status === 'processing' && <Loader2 size={12} className="status-icon processing icon-spin"/>}
                    <span>{task.statusMessage}</span>
                 </div>
              </div>
           </div>
         ))}
      </div>
    </div>
  )};

  const renderHeader = () => (
    <header className="app-header">
      <div className="flex items-center gap-4">
        {state.tasks.length > 0 && (
          <button 
            className="layout-toggle-btn"
            onClick={() => setShowSidebar(!showSidebar)}
            title="Toggle Sidebar"
          >
            {showSidebar ? <ChevronLeft size={20}/> : <ChevronRight size={20} />}
          </button>
        )}
        <div className="brand">
            <Layout className="w-6 h-6 text-blue-600" />
            <span>Image2Drawio</span>
        </div>
        
        <label className="flex items-center gap-2 cursor-pointer ml-4 px-3 py-1 bg-white/50 rounded-full border border-gray-200 hover:bg-white transition-all select-none" title="Ignore cache and force reprocessing">
            <input 
                type="checkbox" 
                checked={isForceMode}
                onChange={(e) => setIsForceMode(e.target.checked)}
                className="w-4 h-4 rounded text-blue-600 focus:ring-blue-500 cursor-pointer"
            />
            <span className="text-xs font-medium text-gray-700">Force Reprocess</span>
        </label>
      </div>

      <div className="header-actions">
        {activeTask?.status === 'completed' && (
           <>
             <button 
                className={`layout-toggle-btn ${showOriginal ? 'text-blue-600' : ''}`}
                onClick={() => setShowOriginal(!showOriginal)}
                title="Toggle Reference Image"
             >
                <div className="flex items-center gap-1">
                    {showOriginal ? <Eye size={18}/> : <EyeOff size={18}/>}
                    <span className="text-sm">Reference</span>
                </div>
             </button>
             <button 
                className={`layout-toggle-btn ${showEditor ? 'text-blue-600' : ''}`}
                onClick={() => setShowEditor(!showEditor)}
                title="Toggle Editor"
             >
                <div className="flex items-center gap-1">
                     <Layout size={18} />
                     <span className="text-sm">Editor</span>
                </div>
             </button>
            </>
        )}
        <span className={`backend-status ${backendStatus.toLowerCase()}`}>
            {backendStatus === 'Online' ? 'Backend Connected' : 'Backend Offline'}
        </span>
      </div>
    </header>
  );

  const renderEmptyState = () => (
    <div className="welcome-screen">
      <div className="upload-card">
         <h2 className="text-2xl font-bold mb-4 text-gray-900">Image2Drawio Converter</h2>
         <p className="text-gray-500 mb-8">Upload multiple images to convert them into editable diagrams.</p>
         
         <div 
            className="drop-zone"
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={() => document.getElementById('fileInput').click()}
         >
            <Upload className="w-12 h-12 text-blue-500 mx-auto mb-4" />
            <p className="text-lg text-gray-700">Click or Drag images here</p>
            <input 
                id="fileInput" 
                type="file" 
                accept="image/*" 
                multiple
                onChange={handleFileChange} 
            />
         </div>
      </div>
    </div>
  );

  const renderProcessingState = (task) => (
      <div className="welcome-screen">
          <div className="image-preview-card">
                <img src={task.previewUrl} alt="Preview" className="preview-img" />
                <p className="file-name">{task.name}</p>
          </div>
          
          <div style={{width: '100%', maxWidth: '400px'}}>
             <div className="progress-container">
                 <div className="flex justify-between mb-2">
                    <span className="text-sm font-medium text-gray-700">{task.statusMessage}</span>
                    <span className="text-sm text-gray-500">{Math.round(task.progress)}%</span>
                 </div>
                 <div className="progress-bar-bg">
                    <div className="progress-bar-fill" style={{ width: `${task.progress}%` }}></div>
                 </div>
             </div>
             {task.status === 'error' && (
                 <div className="text-red-500 text-sm mt-2 text-center">{task.error}</div>
             )}
          </div>
      </div>
  );

  const renderEditorApp = (task) => (
    <div className="split-container">
       {/* 1. Original Image Pane */}
       {showOriginal && (
           <div 
             className="image-pane"
             style={{width: imagePaneWidth}}
             ref={imagePaneRef}
           >
               <div className="image-pane-header">
                   <span>Original Reference (Click to Segment)</span>
               </div>
               <div className="image-viewer">
                   <img 
                       src={task.previewUrl} 
                       alt="Reference" 
                       style={{cursor: 'crosshair'}}
                       onClick={handleImageClick}
                   />
               </div>
           </div>
       )}

       {/* Resizer for Image/Editor split */}
       {showOriginal && showEditor && (
          <div className="resizer" onMouseDown={startResizingImagePane} />
       )}

       {/* 2. Editor Pane */}
       {showEditor && (
           <div className="editor-pane">
               {/* Key is important to force re-mount when switching tasks */}
               <DrawioEditor 
                   key={task.id} 
                   xmlContent={task.resultXml} 
                   command={editorCommand}
                   onCommandExecuted={() => setEditorCommand(null)}
               />
           </div>
       )}
    </div>
  );

  // --- Main Render Decision ---
  const renderMainContent = () => {
      if (state.tasks.length === 0) {
          return renderEmptyState();
      }

      if (!activeTask) return <div>Select a file</div>;

      if (activeTask.status === 'completed' && activeTask.resultXml) {
          return renderEditorApp(activeTask);
      }
      
      return renderProcessingState(activeTask);
  };

  return (
    <>
      {renderHeader()}
      <div className="main-content" style={{flexDirection: 'row'}}> 
        {/* Sidebar */}
        {state.tasks.length > 0 && renderSidebar()}
        
        {/* Resizer for Sidebar */}
        {state.tasks.length > 0 && showSidebar && (
            <div className="resizer" onMouseDown={startResizingSidebar} />
        )}

        {/* Main Area */}
        <main style={{flex: 1, position: 'relative', overflow: 'hidden'}}>
            {renderMainContent()}
        </main>
      </div>
    </>
  );
}

export default App;
