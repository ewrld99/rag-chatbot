import { Suspense, lazy, useEffect, useRef, useState } from "react";
import "./App.css";
import ThemeToggle from "./components/ThemeToggle.jsx";
import { clearAdminSession, isAdminAuthenticated } from "./utils/adminAuth.js";

const PublicLayout = lazy(() => import("./layouts/PublicLayout.jsx"));
const AdminLayout = lazy(() => import("./layouts/AdminLayout.jsx"));
const ChatPage = lazy(() => import("./pages/ChatPage.jsx"));
const AdminLoginPage = lazy(() => import("./pages/AdminLoginPage.jsx"));
const RegisterPage = lazy(() => import("./pages/RegisterPage.jsx"));
const AdminDashboardPage = lazy(() => import("./pages/AdminDashboardPage.jsx"));

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

function RouteFallback() {
    return <div className="app-loading" aria-label="Loading" />;
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
                <Suspense fallback={<RouteFallback />}>
                    <AdminLoginPage />
                </Suspense>
                <ThemeToggle theme={theme} onToggle={toggleTheme} className="theme-toggle-admin" />
            </>
        );
    }

    if (path === "/register") {
        return (
            <>
                <Suspense fallback={<RouteFallback />}>
                    <RegisterPage />
                </Suspense>
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
                <Suspense fallback={<RouteFallback />}>
                    <ProtectedAdminRoute>
                        <AdminLayout>
                            <AdminDashboardPage />
                        </AdminLayout>
                    </ProtectedAdminRoute>
                </Suspense>
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
            <Suspense fallback={<RouteFallback />}>
                <PublicLayout>
                    {publicRoutes[path] ?? <Redirect to="/" />}
                </PublicLayout>
            </Suspense>
            <ThemeToggle theme={theme} onToggle={toggleTheme} />
        </>
    );
}
