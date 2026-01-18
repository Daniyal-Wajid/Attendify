import { Bell } from "lucide-react";

export default function Topbar() {
  return (
    <div className="flex items-center justify-between px-6 py-4 bg-white border-b">
      <input
        placeholder="Search violations, students..."
        className="w-96 px-4 py-2 border rounded-lg text-sm focus:outline-none"
      />

      <div className="flex items-center gap-4">
        <Bell />
        <div className="w-8 h-8 rounded-full bg-blue-600 text-white flex items-center justify-center">
          J
        </div>
        <span className="font-medium">John Admin</span>
      </div>
    </div>
  );
}
