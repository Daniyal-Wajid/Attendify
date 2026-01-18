import { Bell, Eye } from "lucide-react";
import PriorityBadge from "./PriorityBadge";
import { notifications } from "../data/mockNotifications";

export default function NotificationsList() {
  return (
    <div className="bg-white border rounded-xl p-6 space-y-4">
      <h2 className="text-lg font-semibold">All Notifications</h2>

      {notifications.map((n) => (
        <div
          key={n.id}
          className={`flex items-center justify-between border rounded-lg p-4 ${
            !n.read ? "bg-slate-50" : ""
          }`}
        >
          <div className="flex items-center gap-4">
            <div className="p-3 rounded-full bg-red-100 text-red-600">
              <Bell />
            </div>

            <div>
              <p className="font-medium">{n.title}</p>
              {n.violationId && (
                <p className="text-sm text-slate-500">
                  Violation ID: {n.violationId}
                </p>
              )}
            </div>
          </div>

          <div className="flex items-center gap-6">
            <PriorityBadge level={n.priority} />
            <span className="text-sm text-slate-500">{n.time}</span>
            <Eye className="cursor-pointer text-slate-500 hover:text-blue-600" />
          </div>
        </div>
      ))}
    </div>
  );
}
