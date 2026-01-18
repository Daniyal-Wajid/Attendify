import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  AlertTriangle,
  Users,
  Bell,
  Camera,
  BarChart2,
  History,
} from "lucide-react";

const navItems = [
  {
    label: "Dashboard",
    icon: LayoutDashboard,
    to: "/admin/admindashboard",
  },
  {
    label: "Violations",
    icon: AlertTriangle,
    to: "/admin/violations",
  },
  {
    label: "Students",
    icon: Users,
    to: "/admin/students",
  },
  {
    label: "Notifications",
    icon: Bell,
    to: "/admin/notifications",
  },
  {
    label: "Cameras",
    icon: Camera,
    to: "/admin/cameras",
  },
  {
    label: "Analytics",
    icon: BarChart2,
    to: "/admin/analytics",
  },
  {
    label: "History Logs",
    icon: History,
    to: "/admin/history-logs",
  },
];

export default function Sidebar() {
  return (
    <aside className="w-64 bg-slate-900 text-slate-200 flex flex-col">
      {/* Logo */}
      <div className="p-6 text-xl font-bold text-white flex items-center gap-2">
        <span className="bg-blue-600 w-8 h-8 flex items-center justify-center rounded-lg">
          D
        </span>
        DMS
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 px-3">
        {navItems.map(({ label, icon: Icon, to }) => (
          <NavLink
            key={label}
            to={to}
            className={({ isActive }) =>
              `flex items-center gap-3 px-4 py-3 rounded-lg transition
              ${
                isActive
                  ? "bg-blue-600 text-white"
                  : "text-slate-300 hover:bg-slate-800 hover:text-white"
              }`
            }
          >
            <Icon size={18} />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>

      {/* User */}
      <div className="p-4 border-t border-slate-800">
        <div className="text-sm text-white">John Admin</div>
        <div className="text-xs text-slate-400">Admin</div>
      </div>
    </aside>
  );
}
