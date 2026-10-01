import { useRef, useState, useCallback, useEffect } from 'react';
import Webcam from 'react-webcam';
import { postImage } from '../api.js';

export default function HelmetDetect() {
  const webcamRef = useRef(null);
  const [imgSrc, setImgSrc] = useState(null);
  const [detections, setDetections] = useState([]);
  const [summary, setSummary] = useState({});
  const [fps, setFps] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [latency, setLatency] = useState(0);
  const [error, setError] = useState(null);
  const intervalRef = useRef(null);
  const lastTimeRef = useRef(0);
  const frameCountRef = useRef(0);

  const captureAndDetect = useCallback(async () => {
    if (!webcamRef.current) return;
    const imageSrc = webcamRef.current.getScreenshot();
    if (!imageSrc) return;

    try {
      const blob = await fetch(imageSrc).then((r) => r.blob());
      const t0 = performance.now();
      const data = await postImage('/api/detect', blob, { conf: 0.25 });
      const ms = performance.now() - t0;

      setDetections(data.detections || []);
      setSummary(data.summary || {});
      setLatency(data.latency_ms ?? ms);
      setImgSrc(data.image || imageSrc);

      frameCountRef.current++;
      const now = performance.now();
      if (now - lastTimeRef.current >= 1000) {
        setFps(frameCountRef.current);
        frameCountRef.current = 0;
        lastTimeRef.current = now;
      }
      setError(null);
    } catch (err) {
      setError('Lỗi kết nối backend: ' + err.message);
    }
  }, []);

  useEffect(() => {
    if (isRunning) {
      lastTimeRef.current = performance.now();
      frameCountRef.current = 0;
      intervalRef.current = window.setInterval(captureAndDetect, 200);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      setFps(0);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isRunning, captureAndDetect]);

  return (
    <section className="grid">
      <div>
        <h2>🪖 Phát hiện mũ bảo hiểm (Webcam)</h2>
        <p className="muted">YOLO11n fine-tune trên 1.368 ảnh mũ bảo hiểm (Roboflow, CC BY 4.0).</p>
        <label>
          Webcam trực tiếp
          <Webcam
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            videoConstraints={{ width: 640, height: 480 }}
            style={{ width: '100%', border: '1px solid #ccc', borderRadius: 8, marginTop: 8 }}
          />
        </label>
        <div style={{ marginTop: 16, display: 'flex', gap: 16, alignItems: 'center' }}>
          <button
            onClick={() => setIsRunning(!isRunning)}
            style={{
              padding: '8px 24px',
              background: isRunning ? '#e53e3e' : '#38a169',
              color: 'white',
              border: 'none',
              borderRadius: 6,
              fontWeight: 'bold',
              cursor: 'pointer',
            }}
          >
            {isRunning ? '⏹ Dừng' : '▶ Bắt đầu'}
          </button>
          <div><strong>FPS:</strong> {fps}</div>
          <div><strong>Độ trễ:</strong> {latency.toFixed(0)} ms</div>
        </div>
        {error && <p className="error">{error}</p>}
      </div>
      <div>
        <h3>Kết quả phát hiện</h3>
        {imgSrc && <img src={imgSrc} alt="Detected" className="preview" />}
        {Object.keys(summary).length > 0 && (
          <div style={{ marginTop: 12 }}>
            <h4>Thống kê:</h4>
            <ul>
              {Object.entries(summary).map(([label, count]) => (
                <li key={label}>{label}: <strong>{count}</strong></li>
              ))}
            </ul>
          </div>
        )}
        {detections.length > 0 && (
          <div style={{ marginTop: 12 }}>
            <h4>Chi tiết ({detections.length} đối tượng):</h4>
            <table>
              <thead><tr><th>Lớp</th><th>Độ tin cậy</th></tr></thead>
              <tbody>
                {detections.map((d, i) => (
                  <tr key={i}>
                    <td>{d.label}</td>
                    <td>{(d.score * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
}