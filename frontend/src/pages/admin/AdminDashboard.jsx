import Topbar from "../../components/Topbar";
import StatCard from "../../components/StatCard";
import ViolationsTable from "../../components/ViolationsTable";
import { AlertTriangle, Clock, Camera } from "lucide-react";

export default function AdminDashboard() {
  return (
    <>
      <Topbar />

      <div className="p-6 space-y-6 bg-gray-50 dark:bg-gray-900 min-h-full">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Dashboard</h1>
          <p className="text-slate-500 dark:text-gray-400">Welcome back, John Admin</p>
        </div>

        <div className="grid grid-cols-4 gap-6">
          <StatCard
            title="Today's Violations"
            value="1"
            icon={<AlertTriangle />}
            color="bg-blue-100 text-blue-600"
          />
          <StatCard
            title="Unverified"
            value="18"
            icon={<Clock />}
            color="bg-orange-100 text-orange-600"
          />
          <StatCard
            title="High Severity Alerts"
            value="0"
            icon={<AlertTriangle />}
            color="bg-red-100 text-red-600"
          />
          <StatCard
            title="Active Cameras"
            value="4/5"
            icon={<Camera />}
            color="bg-green-100 text-green-600"
          />
        </div>

        <ViolationsTable />
      </div>
    </>
  );
}
