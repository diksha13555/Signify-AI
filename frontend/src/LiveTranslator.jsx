import {
  HandLandmarker,
  FilesetResolver
} from "@mediapipe/tasks-vision";


import { useState, useRef, useEffect } from "react";
import {
  Camera, CameraOff, Delete, Volume2,
  Copy, Trash2, Plus, Check
} from "lucide-react";

const API = "";

export default function LiveTranslator() {
  const video = useRef(null);
  const detector = useRef(null);
  const overlay = useRef(null);
  const animation = useRef(0);
  const pending = useRef(false);
  const lastRequest = useRef(0);
  const handEpoch = useRef(0);
  const starting = useRef(false);
  const canvas = useRef(null);
  const stream = useRef(null);
  const active = useRef(false);
  const generation = useRef(0);
  const stable = useRef({ letter: "", since: 0 });
  const latched = useRef("");
  const autoRef = useRef(true);
  const matches = useRef(0);
  const missing = useRef(0);

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
    cancelAnimationFrame(animation.current);
    handEpoch.current += 1;
    matches.current = 0;
    missing.current = 0;
    if (overlay.current) {
      const o = overlay.current;
      o.getContext("2d").clearRect(0, 0, o.width, o.height);
    }
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

  async function browserTracker() {
    if (detector.current) return;

    const vision = await FilesetResolver.forVisionTasks("/wasm");

    detector.current = await HandLandmarker.createFromOptions(
      vision,
      {
        baseOptions: {
          modelAssetPath: "/models/hand_landmarker.task",
          delegate: "CPU"
        },
        runningMode: "VIDEO",
        numHands: 1,
        minHandDetectionConfidence: 0.3,
        minHandPresenceConfidence: 0.3,
        minTrackingConfidence: 0.4
      }
    );
  }

  async function classify(points, id, epoch) {
    pending.current = true;

    try {
      const response = await fetch("/api/landmarks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ landmarks: points }),
        signal: AbortSignal.timeout(15000)
      });

      if (!response.ok) {
        throw new Error("Recognition API unavailable");
      }

      const data = await response.json();

      if (!active.current ||
          generation.current !== id ||
          handEpoch.current !== epoch) return;

      const detected = data.letter || "";
      const conf = data.confidence || 0;

      setLetter(detected);
      setConfidence(conf);
      setLatency(data.latency_ms || 0);
      setError("");

      if (conf >= 0.80 && detected) {
        if (stable.current.letter === detected) {
          matches.current += 1;
        } else {
          stable.current = {
            letter: detected,
            since: Date.now()
          };
          matches.current = 1;
        }

        if (
          autoRef.current &&
          matches.current >= 3 &&
          Date.now() - stable.current.since >= 600 &&
          latched.current !== detected
        ) {
          setWord(w => w + detected);
          latched.current = detected;
          matches.current = 0;
        }
      } else {
        matches.current = 0;
        stable.current = { letter: "", since: 0 };
      }

    } catch (err) {
      if (active.current && generation.current === id) {
        setError(err.message);
        matches.current = 0;
      }
    } finally {
      pending.current = false;
    }
  }

  function predictLoop(id) {
    let lastFrame = -1;
    let lastDetection = 0;

    const edges = [
      [0,1],[1,2],[2,3],[3,4],
      [0,5],[5,6],[6,7],[7,8],
      [5,9],[9,10],[10,11],[11,12],
      [9,13],[13,14],[14,15],[15,16],
      [13,17],[0,17],[17,18],[18,19],[19,20]
    ];

    function tick(now) {
      if (!active.current || generation.current !== id) return;

      animation.current = requestAnimationFrame(tick);

      const v = video.current;
      const c = canvas.current;
      const o = overlay.current;

      if (!v || !c || !o || v.readyState < 2) return;
      if (v.currentTime === lastFrame || now - lastDetection < 66) return;

      lastFrame = v.currentTime;
      lastDetection = now;

      try {
        const w = v.videoWidth;
        const h = v.videoHeight;
        const size = Math.max(w, h);
        const scale = 224 / size;
        const left = (size - w) / 2;
        const top = (size - h) / 2;

        c.width = 224;
        c.height = 224;

        const input = c.getContext("2d");
        input.fillStyle = "#000";
        input.fillRect(0, 0, 224, 224);

        input.drawImage(
          v,
          left * scale,
          top * scale,
          w * scale,
          h * scale
        );

        const result = detector.current.detectForVideo(c, now);
        const hand = result.landmarks?.[0];

        if (o.width !== w || o.height !== h) {
          o.width = w;
          o.height = h;
        }

        const ctx = o.getContext("2d");
        ctx.clearRect(0, 0, w, h);

        if (!hand) {
          missing.current += 1;
          handEpoch.current += 1;
          matches.current = 0;
          stable.current = { letter: "", since: 0 };

          if (missing.current >= 5) {
            latched.current = "";
            setLetter("");
            setConfidence(0);
          }
          return;
        }

        missing.current = 0;

        const xy = hand.map(p => ({
          x: p.x * size - left,
          y: p.y * size - top
        }));

        ctx.strokeStyle = "#00e5ff";
        ctx.lineWidth = 2.5;

        for (const [a,b] of edges) {
          ctx.beginPath();
          ctx.moveTo(xy[a].x, xy[a].y);
          ctx.lineTo(xy[b].x, xy[b].y);
          ctx.stroke();
        }

        ctx.fillStyle = "#ffdf45";

        for (const p of xy) {
          ctx.beginPath();
          ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
          ctx.fill();
        }

        const xs = xy.map(p => p.x);
        const ys = xy.map(p => p.y);

        ctx.strokeRect(
          Math.min(...xs) - 10,
          Math.min(...ys) - 10,
          Math.max(...xs) - Math.min(...xs) + 20,
          Math.max(...ys) - Math.min(...ys) + 20
        );

        if (!pending.current &&
            now - lastRequest.current >= 250) {
          lastRequest.current = now;

          classify(
            hand.map(p => [p.x, p.y, p.z]),
            id,
            handEpoch.current
          );
        }
      } catch (err) {
        setError(err.message);
        stopCamera();
      }
    }

    animation.current = requestAnimationFrame(tick);
  }

  async function startCamera() {
    if (active.current || starting.current) return;
    starting.current = true;

    try {
      setError("");

      await browserTracker();

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

          <div className="camera-frame" style={{ position: "relative" }}>
            <video
              ref={video}
              muted
              playsInline
              autoPlay
              style={{
                visibility: running ? "visible" : "hidden",
                objectFit: "contain",
                width: "100%",
                height: "100%",
                transform: "none"
              }}
            />

            {!running && (
              <div className="camera-placeholder">
                <Camera size={48} />
                <h3>Ready to translate?</h3>
                <p>Enable your camera to begin.</p>
              </div>
            )}

            {false && running && processed && (
              <img
                className="tracking-preview"
                src={processed}
                alt="Latest hand tracking result"
              />
            )}

            <canvas ref={canvas} hidden />

            <canvas
              ref={overlay}
              style={{
                position: "absolute",
                inset: 0,
                width: "100%",
                height: "100%",
                objectFit: "contain",
                pointerEvents: "none",
                zIndex: 2
              }}
            />
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
