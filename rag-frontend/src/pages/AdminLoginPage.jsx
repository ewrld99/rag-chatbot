import { useEffect } from "react";
import LoginForm from "../components/admin/loginForm.jsx";
import { loginUser } from "../api/chatApi.js";
import { clearAdminSession, isAdminAuthenticated, saveAdminSession } from "../utils/adminAuth.js";

function navigate(to) {
    window.history.replaceState({}, "", to);
    window.dispatchEvent(new PopStateEvent("popstate"));
}

export default function AdminLoginPage() {
    useEffect(() => {
        if (isAdminAuthenticated()) {
            navigate("/admin/dashboard");
        }
    }, []);

    const handleLogin = async (username, password) => {
        const user = await loginUser(username, password);

        if (user.role === "admin") {
            window.localStorage.removeItem("ragUser");
            saveAdminSession(user);
            navigate("/admin/dashboard");
            return;
        }

        clearAdminSession();
        window.localStorage.setItem("ragUser", JSON.stringify(user));
        navigate("/chat");
    };

    return (
        <section className="admin-login-page" aria-labelledby="admin-login-title">
            <div className="admin-login-shell">
                <LoginForm onLogin={handleLogin} />
            </div>
        </section>
    );
}
