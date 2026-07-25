import { useEffect, useRef, useState } from "react";
import "./App.css";
import PublicLayout from "./layouts/PublicLayout.jsx";
import AdminLayout from "./layouts/AdminLayout.jsx";
import ChatPage from "./pages/ChatPage.jsx";
import AdminLoginPage from "./pages/AdminLoginPage.jsx";
import RegisterPage from "./pages/RegisterPage.jsx";
import AdminDashboardPage from "./pages/AdminDashboardPage.jsx";
import ThemeToggle from "./components/ThemeToggle.jsx";
import { clearAdminSession, isAdminAuthenticated } from "./utils/adminAuth.js";

function normalizePath(pathname) {
    if (pathname.length > 1 && pathname.endsWith("/")) {
        return pathname.slice(0, -1);
    }

    return pathname || "/";
}

function Redirect({ to, replace = true }) {
    useEffect(() => {
        window.history[replace ? "replaceState" : "pushState"]({}, "", to);
        window.dispatchEvent(new PopStateEvent("popstate"));
    }, [replace, to]);

    return null;
}

function ProtectedAdminRoute({ children }) {
    if (!isAdminAuthenticated()) {
        return <Redirect to="/login" />;
    }

    return children;
}

export default function App() {
    const [path, setPath] = useState(() => normalizePath(window.location.pathname));
    const pathRef = useRef(path);
    const [theme, setTheme] = useState(() => {
        const savedTheme = window.localStorage.getItem("udomTheme");
        if (savedTheme === "dark" || savedTheme === "light") return savedTheme;
        return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    });

    useEffect(() => {
        pathRef.current = path;
    }, [path]);

    useEffect(() => {
        const handlePopState = () => {
            const nextPath = normalizePath(window.location.pathname);

            if (pathRef.current === "/admin/dashboard" && nextPath !== "/admin/dashboard") {
                clearAdminSession();
            }

            setPath(nextPath);
        };

        window.addEventListener("popstate", handlePopState);

        return () => window.removeEventListener("popstate", handlePopState);
    }, []);

    useEffect(() => {
        document.documentElement.dataset.theme = theme;
        window.localStorage.setItem("udomTheme", theme);
    }, [theme]);

    const toggleTheme = () => setTheme((current) => (current === "dark" ? "light" : "dark"));

    if (path === "/login") {
        return (
            <>
                <AdminLoginPage />
                <ThemeToggle theme={theme} onToggle={toggleTheme} className="theme-toggle-admin" />
            </>
        );
    }

    if (path === "/register") {
        return (
            <>
                <RegisterPage />
                <ThemeToggle theme={theme} onToggle={toggleTheme} />
            </>
        );
    }

    if (path === "/admin/login") {
        return <Redirect to="/login" />;
    }

    if (path === "/admin/dashboard") {
        return (
            <>
                <ProtectedAdminRoute>
                    <AdminLayout>
                        <AdminDashboardPage />
                    </AdminLayout>
                </ProtectedAdminRoute>
                <ThemeToggle theme={theme} onToggle={toggleTheme} />
            </>
        );
    }

    const publicRoutes = {
        "/": <ChatPage />,
        "/chat": <ChatPage />,
    };

    return (
        <>
            <PublicLayout>
                {publicRoutes[path] ?? <Redirect to="/" />}
            </PublicLayout>
            <ThemeToggle theme={theme} onToggle={toggleTheme} />
        </>
    );
}
