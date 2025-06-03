import React, { useState } from "react";

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [caption, setCaption] = useState("");
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setFile(file);
      setPreview(URL.createObjectURL(file));
      setCaption("");
    }
  };

  const handleUpload = async () => {
    if (!file) return alert("请先选择一张图片哦~");
    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error("网络错误");
      const data = await res.json();
      console.log(data.caption);
    } catch (e) {
      alert("请求失败：" + e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 600, margin: "auto", padding: 20, fontFamily: "Arial" }}>
      <h2>图像描述生成器</h2>
      <input type="file" accept="image/*" onChange={handleFileChange} />
      {preview && (
        <div style={{ marginTop: 10 }}>
          <img src={preview} alt="preview" style={{ maxWidth: "100%" }} />
        </div>
      )}
      <button
        onClick={handleUpload}
        disabled={loading}
        style={{ marginTop: 15, padding: "10px 20px", cursor: "pointer" }}
      >
        {loading ? "生成中..." : "生成描述"}
      </button>
      {caption && (
        <div style={{ marginTop: 20, fontSize: 18, color: "#333" }}>
          <strong>描述：</strong> {caption}
        </div>
      )}
    </div>
  );
}
