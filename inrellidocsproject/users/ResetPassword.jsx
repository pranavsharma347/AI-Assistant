import React, { useEffect, useState } from "react";
import { useParams, Link, useNavigate } from "react-router-dom";
import axios from "axios";

function ResetPassword() {
  const { uidb64, token } = useParams();
  const navigate = useNavigate();

  const [password, setPassword] = useState("");
  const [password2, setPassword2] = useState("");

  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [validToken, setValidToken] = useState(false);

  // 🎨 Dark Theme
  const bg = "#0D0D0D";
  const cardBg = "#1A1A1A";
  const text = "#FFFFFF";
  const muted = "#BBBBBB";
  const border = "#333";

  // 🔐 CHECK TOKEN
  useEffect(() => {
    const verifyToken = async () => {
      try {
        await axios.get(
          `http://localhost:8000/user/password-reset/${uidb64}/${token}/`
        );
        setValidToken(true);
      } catch {
        setError("❌ Reset link is invalid or expired");
      }
    };

    verifyToken();
  }, [uidb64, token]);

  // 🔁 RESET PASSWORD
  const handleResetPassword = async () => {
    setError("");
    setMessage("");

    if (!password || !password2) {
      setError("❌ Both password fields are required");
      return;
    }

    if (password.length < 8) {
      setError("❌ Password must be at least 8 characters");
      return;
    }

    if (password !== password2) {
      setError("❌ Passwords do not match");
      return;
    }

    setLoading(true);

    try {
      await axios.put(
        "http://localhost:8000/user/password-reset-complete",
        {
          password,
          password2,
          uidb64,
          token,
        }
      );

      setMessage("✅ Password reset successful! You can now login.");

      setTimeout(() => {
        navigate("/login");
      }, 2000);

    } catch {
      setError("❌ Failed to reset password. Try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        backgroundColor: bg,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 20,
      }}
    >
      <div
        className="shadow-lg rounded-4 p-5 w-100"
        style={{
          maxWidth: 420,
          backgroundColor: cardBg,
          border: `1px solid ${border}`,
        }}
      >
        <h3 className="fw-bold mb-2" style={{ color: text }}>
          Reset Password 🔐
        </h3>

        <p className="mb-4" style={{ color: muted, fontSize: 14 }}>
          IntelliDocs — Create a new password for your account
        </p>

        {/* ERROR */}
        {error && (
          <div className="alert alert-danger py-2 text-center">
            {error}
          </div>
        )}

        {/* SUCCESS */}
        {message && (
          <div className="alert alert-success py-2 text-center">
            {message}
          </div>
        )}

        {/* FORM */}
        {validToken && !message && (
          <>
            <input
              type="password"
              placeholder="New password"
              className="form-control mb-3"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />

            <input
              type="password"
              placeholder="Confirm new password"
              className="form-control mb-3"
              value={password2}
              onChange={(e) => setPassword2(e.target.value)}
            />

            <button
              onClick={handleResetPassword}
              disabled={loading}
              className="btn w-100 fw-bold mb-3"
              style={{
                background: "linear-gradient(135deg, #FF7B00, #FF5100)",
                color: "#fff",
                borderRadius: 30,
                opacity: loading ? 0.7 : 1,
              }}
            >
              {loading ? "Resetting..." : "Reset Password"}
            </button>
          </>
        )}

        {/* BACK TO LOGIN */}
        <div className="text-center mt-3">
          <Link to="/login" style={{ color: "#FF7B00", fontSize: 14 }}>
            ← Back to Login
          </Link>
        </div>
      </div>
    </div>
  );
}

export default ResetPassword;