import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

function App() {
  const [imageFile, setImageFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [repredicting, setRepredicting] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [editableKeypoints, setEditableKeypoints] = useState([]);
  const [dragIndex, setDragIndex] = useState(null);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const imageRef = useRef(null);

  const previewUrl = useMemo(() => {
    if (!imageFile) return "";
    return URL.createObjectURL(imageFile);
  }, [imageFile]);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const updatePointFromClient = (index, clientX, clientY) => {
    const imageEl = imageRef.current;
    if (!imageEl || imageSize.width <= 0 || imageSize.height <= 0) return;

    const rect = imageEl.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;

    const rawX = ((clientX - rect.left) / rect.width) * imageSize.width;
    const rawY = ((clientY - rect.top) / rect.height) * imageSize.height;

    const x = Math.max(0, Math.min(imageSize.width, rawX));
    const y = Math.max(0, Math.min(imageSize.height, rawY));

    setEditableKeypoints((prev) =>
      prev.map((point, pointIndex) => (pointIndex === index ? [x, y] : point))
    );
  };

  useEffect(() => {
    if (dragIndex === null) return undefined;

    const onMove = (event) => {
      updatePointFromClient(dragIndex, event.clientX, event.clientY);
    };

    const onUp = () => {
      setDragIndex(null);
    };

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);

    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, [dragIndex, imageSize.width, imageSize.height]);

  const onSubmit = async (event) => {
    event.preventDefault();
    if (!imageFile) {
      setError("Please choose an image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("image", imageFile);

      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.detail || "Prediction request failed.");
      }

      const data = await response.json();
      setResult(data);
      setEditableKeypoints(Array.isArray(data.keypoints) ? data.keypoints : []);
    } catch (err) {
      setError(err.message || "Unexpected error");
    } finally {
      setLoading(false);
    }
  };

  const onRePredict = async () => {
    if (!result || !Array.isArray(editableKeypoints) || editableKeypoints.length === 0) {
      setError("No keypoints available. Run an initial prediction first.");
      return;
    }

    setRepredicting(true);
    setError("");

    try {
      const response = await fetch(`${API_BASE}/predict/keypoints`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          keypoints: editableKeypoints,
          age: result.age,
          gender: result.gender,
        }),
      });

      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.detail || "Adjusted-keypoint prediction request failed.");
      }

      const data = await response.json();
      setResult((prev) => ({ ...prev, ...data }));
      setEditableKeypoints(Array.isArray(data.keypoints) ? data.keypoints : editableKeypoints);
    } catch (err) {
      setError(err.message || "Unexpected error");
    } finally {
      setRepredicting(false);
    }
  };

  return (
    <main className="app">
      <section className="card">
        <h1>C-Spine Degeneration Predictor</h1>
        <p className="subtitle">Upload an X-ray image and run your updated inference pipeline.</p>
        <p className="subtitle">
          Filename rule: gender is the 5th character, age starts at the 6th character.
          Example: <code>0001054.png</code> means gender=0, age=54.
        </p>

        <form onSubmit={onSubmit} className="form">
          <label>
            Image
            <input
              type="file"
              accept="image/*"
              onChange={(e) => {
                setImageFile(e.target.files?.[0] ?? null);
                setResult(null);
                setEditableKeypoints([]);
                setError("");
                setImageSize({ width: 0, height: 0 });
              }}
            />
          </label>

          <button type="submit" disabled={loading}>
            {loading ? "Predicting..." : "Predict"}
          </button>
        </form>

        {previewUrl && !result && (
          <div className="preview">
            <h3>Selected X-ray</h3>
            <img src={previewUrl} alt="X-ray preview" />
          </div>
        )}

        {error && <p className="error">{error}</p>}

        {result && (
          <section className="result">
            <h2>Result</h2>
            <p>
              Label: <strong>{result.label}</strong>
            </p>
            <p>
              Prediction: <strong>{result.prediction}</strong>
            </p>
            <p>
              Probability: <strong>{(result.probability * 100).toFixed(2)}%</strong>
            </p>
            <p>
              Parsed age: <strong>{result.age}</strong>
            </p>
            <p>
              Parsed gender: <strong>{result.gender}</strong>
            </p>
            <p>
              Keypoints detected: <strong>{Array.isArray(result.keypoints) ? result.keypoints.length : 0}</strong>
            </p>

            {previewUrl && editableKeypoints.length > 0 && (
              <div className="keypoint-editor">
                <h3>Adjust Keypoints</h3>
                <p className="subtitle">Drag any point to correct placement, then run prediction again.</p>
                <div className="editor-stage">
                  <img
                    ref={imageRef}
                    src={previewUrl}
                    alt="X-ray for keypoint adjustment"
                    className="keypoint-image"
                    onLoad={(event) => {
                      setImageSize({
                        width: event.currentTarget.naturalWidth,
                        height: event.currentTarget.naturalHeight,
                      });
                    }}
                  />
                  {imageSize.width > 0 && imageSize.height > 0 && (
                    <svg
                      className="keypoint-svg"
                      viewBox={`0 0 ${imageSize.width} ${imageSize.height}`}
                      preserveAspectRatio="none"
                    >
                      {editableKeypoints.slice(0, -1).map((point, index) => (
                        <line
                          key={`link-${index}`}
                          x1={point[0]}
                          y1={point[1]}
                          x2={editableKeypoints[index + 1][0]}
                          y2={editableKeypoints[index + 1][1]}
                          className="keypoint-link"
                        />
                      ))}
                      {editableKeypoints.map((point, index) => (
                        <g key={`kp-${index}`}>
                          <circle
                            cx={point[0]}
                            cy={point[1]}
                            r="7"
                            className="keypoint-circle"
                            onPointerDown={(event) => {
                              event.preventDefault();
                              setDragIndex(index);
                            }}
                          />
                          <text x={point[0] + 9} y={point[1] - 8} className="keypoint-text">
                            {index + 1}
                          </text>
                        </g>
                      ))}
                    </svg>
                  )}
                </div>

                <button
                  type="button"
                  onClick={onRePredict}
                  disabled={loading || repredicting || editableKeypoints.length === 0}
                >
                  {repredicting ? "Predicting with adjusted keypoints..." : "Predict Again with Adjusted Keypoints"}
                </button>
              </div>
            )}

            <h3>Extracted Features</h3>
            <div className="features">
              {Object.entries(result.features || {}).map(([key, value]) => (
                <div key={key} className="feature">
                  <span>{key}</span>
                  <strong>{Number.isFinite(value) ? Number(value).toFixed(5) : String(value)}</strong>
                </div>
              ))}
            </div>
          </section>
        )}
      </section>
    </main>
  );
}

export default App;
