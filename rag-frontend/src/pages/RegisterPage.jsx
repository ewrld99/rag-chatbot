import RegisterForm from "../components/admin/RegisterForm.jsx";
import { registerUser } from "../api/chatApi.js";
import { clearAdminSession, saveAdminSession } from "../utils/adminAuth.js";

function navigate(to) {
    window.history.replaceState({}, "", to);
    window.dispatchEvent(new PopStateEvent("popstate"));
}

export default function RegisterPage() {
    const handleRegister = async (username, password, registrationNumber, programme, campus, admissionYear) => {
        const user = await registerUser(username, password, registrationNumber, programme, campus, admissionYear);

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
        <section className="admin-login-page" aria-labelledby="register-title">
            <div className="admin-login-shell">
                <RegisterForm onRegister={handleRegister} onLoginClick={() => navigate("/login")} />
            </div>
        </section>
    );
}
