
import { useState, useRef, useEffect } from "react";
import {
  Camera, CameraOff, Delete, Volume2,
  Copy, Trash2, Plus, Check
} from "lucide-react";

const API = "";

export default function LiveTranslator() {
  const video = useRef(null);
  const canvas = useRef(null);
  const stream = useRef(null);
  const active = useRef(false);
  const generation = useRef(0);
  const stable = useRef({ letter: "", since: 0 });
  const latched = useRef("");
  const autoRef = useRef(true);

  const [running, setRunning] = useState(false);
  const [letter, setLetter] = useState("");
  const [confidence, setConfidence] = useState(0);
  const [processed, setProcessed] = useState("");
  const [word, setWord] = useState("");
  const [message, setMessage] = useState("");
  const [latency, setLatency] = useState(0);
  const [auto, setAuto] = useState(true);
  const [error, setError] = useState("");

  function stopCamera() {
    active.current = false;
    generation.current += 1;
    stream.current?.getTracks().forEach(t => t.stop());
    stream.current = null;
    setRunning(false);
    setProcessed("");
    setLetter("");
    setConfidence(0);
    stable.current = { letter: "", since: 0 };
    latched.current = "";
  }

  useEffect(() => {
    return () => {
      active.current = false;
      generation.current += 1;
      stream.current?.getTracks().forEach(t => t.stop());
    };
  }, []);

  async function predictLoop(id) {
    while (active.current && generation.current === id) {
      try {
        const v = video.current;
        const c = canvas.current;

        if (v && c && v.readyState >= 2) {
          c.width = 320;
          c.height = Math.round(
            320 * v.videoHeight / v.videoWidth
          );

          c.getContext("2d").drawImage(
            v, 0, 0, c.width, c.height
          );

          const blob = await new Promise(resolve =>
            c.toBlob(resolve, "image/jpeg", 0.65)
          );

          if (!blob) continue;

          const form = new FormData();
          form.append("file", blob, "frame.jpg");

          const response = await fetch(
            `${API}/api/predict`,
            {
              method: "POST",
              body: form,
              signal: AbortSignal.timeout(10000)
            }
          );

          if (!response.ok) {
            throw new Error("Prediction request failed");
          }

          const data = await response.json();

          if (!active.current ||
              generation.current !== id) return;

          const detected = data.letter || "";

          setLetter(detected);
          setConfidence(data.confidence || 0);
          setProcessed(data.image || "");
          setLatency(data.latency_ms || 0);
          setError("");

          if (!detected) {
            latched.current = "";
            stable.current = { letter: "", since: 0 };
          } else if (data.confidence >= 0.65) {
            if (stable.current.letter !== detected) {
              stable.current = {
                letter: detected,
                since: Date.now()
              };
            }

            if (
              autoRef.current &&
              Date.now() - stable.current.since >= 900 &&
              latched.current !== detected
            ) {
              setWord(w => w + detected);
              latched.current = detected;
            }
          } else {
            stable.current = { letter: "", since: 0 };
          }
        }
      } catch (err) {
        if (generation.current !== id) return;
        setError(err.message);
        await new Promise(r => setTimeout(r, 500));
      }

      await new Promise(r => setTimeout(r, 180));
    }
  }

  async function startCamera() {
    if (active.current) return;

    try {
      setError("");

      const media = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 24, max: 30 }
        },
        audio: false
      });

      stream.current = media;
      video.current.srcObject = media;
      await video.current.play();

      active.current = true;
      setRunning(true);

      const id = ++generation.current;
      predictLoop(id);
    } catch (err) {
      stopCamera();
      setError(err.message);
    }
  }

  function endWord() {
    if (!word) return;
    setMessage(m => `${m} ${word}`.trim());
    setWord("");
  }

  function fullText() {
    return `${message} ${word}`.trim();
  }

  function speak() {
    if (!fullText()) return;
    speechSynthesis.cancel();
    speechSynthesis.speak(
      new SpeechSynthesisUtterance(fullText())
    );
  }

  return (
    <section id="translator" className="section">
      <div className="eyebrow">REAL-TIME INTELLIGENCE</div>
      <h2>Live translation studio</h2>
      <p className="muted">
        Your hands become words. Your words become a voice.
      </p>

      <div className="translator-grid">
        <div className="panel">
          <div className="panel-heading">
            <span>Camera workspace</span>
            <span className="status">
              {running ? "● LIVE" : "○ OFFLINE"}
            </span>
          </div>

          <div className="camera-frame">
            <video
              ref={video}
              muted
              playsInline
              autoPlay
              style={{
                visibility: running ? "visible" : "hidden"
              }}
            />

            {!running && (
              <div className="camera-placeholder">
                <Camera size={48} />
                <h3>Ready to translate?</h3>
                <p>Enable your camera to begin.</p>
              </div>
            )}

            {running && processed && (
              <img
                className="tracking-preview"
                src={processed}
                alt="Latest hand tracking result"
              />
            )}

            <canvas ref={canvas} hidden />
          </div>

          <div className="controls">
            <button
              className="primary"
              onClick={running ? stopCamera : startCamera}
            >
              {running ? <CameraOff /> : <Camera />}
              {running ? "Stop camera" : "Start camera"}
            </button>

            <span className="muted">
              {latency.toFixed(0)} ms inference
            </span>
          </div>

          <label className="toggle">
            <input
              type="checkbox"
              checked={auto}
              onChange={e => {
                setAuto(e.target.checked);
                autoRef.current = e.target.checked;
              }}
            />
            Automatically add stable letters
          </label>

          {error && <p className="error">{error}</p>}
        </div>

        <div className="panel prediction-panel">
          <div className="eyebrow">DETECTED ALPHABET</div>

          <div className="detected-letter">
            {letter || "—"}
          </div>

          <div className="panel-heading">
            <span>Model confidence</span>
            <strong>{(confidence * 100).toFixed(1)}%</strong>
          </div>

          <div className="progress">
            <div style={{ width: `${confidence * 100}%` }} />
          </div>

          <div className="word-box">
            <small>CURRENT WORD</small>
            <h3>{word || "Start signing..."}</h3>
          </div>

          <div className="controls">
            <button
              disabled={!letter}
              onClick={() => {
                setWord(w => w + letter);
                latched.current = letter;
              }}
            >
              <Plus /> Add letter
            </button>

            <button className="primary" onClick={endWord}>
              <Check /> End word
            </button>
          </div>

          <div className="word-box">
            <small>YOUR MESSAGE</small>
            <p>{message || "Your completed words appear here."}</p>
          </div>

          <div className="controls">
            <button title="Backspace" onClick={() => {
              if (word) setWord(w => w.slice(0, -1));
              else setMessage(m => m.slice(0, -1));
            }}>
              <Delete />
            </button>

            <button title="Speak" onClick={speak}>
              <Volume2 />
            </button>

            <button title="Copy" onClick={async () => {
              try {
                await navigator.clipboard.writeText(fullText());
              } catch {
                setError("Unable to copy text.");
              }
            }}>
              <Copy />
            </button>

            <button title="Clear" onClick={() => {
              setWord("");
              setMessage("");
            }}>
              <Trash2 />
            </button>
          </div>
        </div>
      </div>

      <p className="muted">
        Hold a gesture for one second. Remove your hand
        before repeating the same letter.
      </p>
    </section>
  );
}
