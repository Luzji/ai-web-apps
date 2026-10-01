import { useEffect, useState } from 'react';
import { postImage } from '../api.js';
import ImagePicker from './ImagePicker.jsx';

export default function Classify() {
  const [file, setFile] = useState(null);
  const [explain, setExplain] = useState(false); // bật Grad-CAM: gọi /api/classify/explain thay vì /api/classify
  const [state, setState] = useState({ status: 'idle' });

  // Chạy lại khi đổi ảnh hoặc bật/tắt Grad-CAM; `cancelled` bỏ kết quả của request cũ nếu người dùng đổi nhanh.
  useEffect(() => {
    if (!file) return undefined;
    let cancelled = false;
    setState({ status: 'loading' });
    postImage(explain ? '/api/classify/explain' : '/api/classify', file, { top_k: 3 })
      .then((data) => { if (!cancelled) setState({ status: 'ok', data }); })
      .catch((err) => { if (!cancelled) setState({ status: 'error', error: err.message }); });
    return () => { cancelled = true; };
  }, [file, explain]);

  return (
    <section className="grid">
      <div>
        <h2>Phân loại món ăn Việt</h2>
        <p className="muted">ResNet-18 fine-tune trên 5 món: phở, bún bò Huế, cơm tấm, bánh mì, hủ tiếu.</p>
        <ImagePicker onChange={setFile} />
        <label className="check">
          <input type="checkbox" checked={explain} onChange={(e) => setExplain(e.target.checked)} />
          Giải thích bằng Grad-CAM (vùng mô hình chú ý)
        </label>
      </div>
      <div>
        {state.status === 'loading' && <p>{explain ? 'Đang dự đoán và tạo bản đồ nhiệt…' : 'Đang dự đoán…'}</p>}
        {state.status === 'error' && <p className="error">{state.error}</p>}
        {state.status === 'ok' && (
          <>
            {!state.data.confident && <p className="warn">Mô hình không chắc chắn — ảnh có thể không thuộc 5 loài đã học.</p>}
            {state.data.predictions.map((p) => (
              <div key={p.label} className="bar">
                <span>{p.label}</span>
                <div className="track"><div className="fill" style={{ width: `${p.score * 100}%` }} /></div>
                <span>{(p.score * 100).toFixed(1)}%</span>
              </div>
            ))}
            {state.data.overlay && (
              <figure>
                <img src={state.data.overlay} alt={`Grad-CAM cho nhãn ${state.data.target}`} className="preview" />
                <figcaption>
                  Vùng đỏ/vàng ảnh hưởng nhiều nhất đến nhãn “{state.data.target}”. Chỉ hiển thị phần giữa ảnh (224×224) mà mô hình
                  thực sự nhìn. Bản đồ cho biết mô hình dựa vào đâu, không chứng minh dự đoán đúng.
                </figcaption>
              </figure>
            )}
            <p className="muted">{state.data.latency_ms} ms</p>
          </>
        )}
      </div>
    </section>
  );
}
