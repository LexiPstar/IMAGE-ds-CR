import React, { useState } from "react";

export default function BatchUpload() {
  const [files, setFiles] = useState([]);
  const [captions, setCaptions] = useState({});
  const [progress, setProgress] = useState({}); // 文件名->上传进度（0~100）

  const handleFilesChange = (e) => {
    const selectedFiles = Array.from(e.target.files);
    setFiles(selectedFiles);
    setCaptions({});
    setProgress({});
  };

  const uploadFile = async (file) => {
    const formData = new FormData();
    formData.append("file", file);

    try {
      setProgress((p) => ({ ...p, [file.name]: 0 }));

      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("上传失败");

      // 模拟进度（因为 fetch 没法原生监听上传进度，实际项目建议用 axios + onUploadProgress）
      for (let i = 0; i <= 100; i += 20) {
        await new Promise((r) => setTimeout(r, 100));
        setProgress((p) => ({ ...p, [file.name]: i }));
      }

      const data = await res.json();
      setCaptions((c) => ({ ...c, [file.name]: data.caption }));
      setProgress((p) => ({ ...p, [file.name]: 100 }));
    } catch (e) {
      setCaptions((c) => ({ ...c, [file.name]: "上传或解析失败" }));
      setProgress((p) => ({ ...p, [file.name]: 0 }));
    }
  };

  const handleUploadAll = () => {
    files.forEach(uploadFile);
  };

  return (
    <div style={{ maxWidth: 700, margin: "auto", padding: 20 }}>
      <h2>批量上传图片生成描述</h2>
      <input type="file" multiple accept="image/*" onChange={handleFilesChange} />
      {files.length > 0 && (
        <button onClick={handleUploadAll} style={{ marginTop: 10 }}>
          一键生成所有描述
        </button>
      )}
      <div style={{ marginTop: 20 }}>
        {files.map((file) => (
          <div
            key={file.name}
            style={{
              marginBottom: 20,
              padding: 10,
              border: "1px solid #ddd",
              borderRadius: 6,
            }}
          >
            <strong>{file.name}</strong>
            <div style={{ height: 8, background: "#eee", borderRadius: 4, marginTop: 5 }}>
              <div
                style={{
                  width: `${progress[file.name] || 0}%`,
                  height: "100%",
                  backgroundColor: progress[file.name] === 100 ? "green" : "#2196f3",
                  borderRadius: 4,
                  transition: "width 0.3s ease",
                }}
              ></div>
            </div>
            <div style={{ marginTop: 8, color: "#555" }}>
              {captions[file.name] || "等待上传..."}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
