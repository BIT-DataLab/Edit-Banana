import React, { useRef, useEffect, useState } from 'react';

const DrawioEditor = ({ xmlContent, command, onCommandExecuted }) => {
  const iframeRef = useRef(null);
  const [isReady, setIsReady] = useState(false);

  // 初始化监听
  useEffect(() => {
    const handleMessage = (event) => {
      // 可以在这里验证 event.origin，但在本地开发时可能是 localhost
      if (!event.data || typeof event.data !== 'string') return;
      
      try {
        const msg = JSON.parse(event.data);
        // console.log("Draw.io Message Received:", msg); // Debug log (too noisy)
        
        if (msg.event === 'init') {
          console.log("Draw.io Init received, sending load...");
          setIsReady(true);
        }
        
        // 处理保存/导出等其他事件 (如果需要)
        // if (msg.event === 'save') { ... }
        
      } catch (e) {
        // 忽略非 JSON 消息
      }
    };

    window.addEventListener('message', handleMessage);
    return () => window.removeEventListener('message', handleMessage);
  }, []);

  // 当 XML 内容变化且 iframe 准备好时，加载内容
  useEffect(() => {
    if (isReady && xmlContent && iframeRef.current) {
        // 构造 load 消息
        // 参考: https://www.drawio.com/doc/faq/embed-mode
        const msg = {
            action: 'load',
            xml: xmlContent,
            autosave: 0
        };
        iframeRef.current.contentWindow.postMessage(JSON.stringify(msg), '*');
    }
  }, [isReady, xmlContent]);

  // 处理外部指令 (Merge, Add etc)
  useEffect(() => {
    if (isReady && command && iframeRef.current) {
        console.log("Draw.io Command:", command);
        iframeRef.current.contentWindow.postMessage(JSON.stringify(command), '*');
        if (onCommandExecuted) {
            onCommandExecuted();
        }
    }
  }, [isReady, command, onCommandExecuted]);

  return (
    <div style={{ width: '100%', height: '100%', border: 'none' }}>
      <iframe
        ref={iframeRef}
        src="https://embed.diagrams.net/?embed=1&spin=1&proto=json&libraries=1"
        title="Draw.io Editor"
        style={{ width: '100%', height: '100%', border: 'none', display: 'block' }}
      />
    </div>
  );
};

export default DrawioEditor;
