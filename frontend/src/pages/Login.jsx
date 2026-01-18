import { useState } from "react";
import { useNavigate } from "react-router-dom";

export default function Login() {
  const [form, setForm] = useState({ email: "", password: "" });
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const toggleDark = () => {
    document.documentElement.classList.toggle("dark");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const res = await fetch("http://localhost:5000/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });

      const data = await res.json();

      if (!res.ok) {
        alert(data.error || "Login failed");
        return;
      }

      localStorage.setItem("token", data.token);
      localStorage.setItem("user", JSON.stringify(data.user));

      if (data.user.role === "admin") navigate("/admin");
      else if (data.user.role === "teacher") navigate("/teacher");
      else navigate("/student");
    } catch {
      alert("Login failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br 
      from-indigo-100 to-blue-200 dark:from-gray-900 dark:to-gray-800 p-4">

      <div className="w-full max-w-md bg-white dark:bg-gray-900 
        rounded-2xl shadow-2xl p-8 border dark:border-gray-700">

        {/* Header */}
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-3xl font-extrabold text-indigo-600 dark:text-indigo-400">
            Attendify
          </h1>
          <button
            onClick={toggleDark}
            className="text-sm text-gray-500 dark:text-gray-300 hover:underline"
          >
            Toggle Theme
          </button>
        </div>

        <p className="text-gray-600 dark:text-gray-400 mb-8">
          Smart Attendance. Powered by AI.
        </p>

        <form onSubmit={handleSubmit} className="space-y-5">
          <input
            type="email"
            required
            placeholder="Email address"
            value={form.email}
            onChange={(e) => setForm({ ...form, email: e.target.value })}
            className="w-full px-4 py-3 rounded-lg border 
              bg-gray-50 dark:bg-gray-800 
              border-gray-300 dark:border-gray-700
              focus:ring-2 focus:ring-indigo-500 outline-none"
          />

          <input
            type="password"
            required
            placeholder="Password"
            value={form.password}
            onChange={(e) => setForm({ ...form, password: e.target.value })}
            className="w-full px-4 py-3 rounded-lg border 
              bg-gray-50 dark:bg-gray-800 
              border-gray-300 dark:border-gray-700
              focus:ring-2 focus:ring-indigo-500 outline-none"
          />

          <button
            type="submit"
            disabled={loading}
            className="w-full py-3 rounded-lg font-semibold text-white
              bg-indigo-600 hover:bg-indigo-700
              transition disabled:opacity-50"
          >
            {loading ? "Signing in..." : "Sign In"}
          </button>
        </form>

        <div className="mt-6 text-center">
          <p className="text-gray-500 dark:text-gray-400 mb-2">
            New student?
          </p>
          <button
            onClick={() => navigate("/register")}
            className="text-indigo-600 dark:text-indigo-400 font-medium hover:underline"
          >
            Create an account
          </button>
        </div>
      </div>
    </div>
  );
}
