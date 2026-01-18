import {
  historySummary,
  activityLogs,
} from "../../data/mockHistoryLogs";
import HistoryStatCard from "../../components/HistoryStatCard";
import TimelineItem from "../../components/TimelineItem";

export default function HistoryLogs() {
  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold">History Logs</h1>
        <p className="text-slate-500">
          View all system activity logs
        </p>
      </div>

      {/* Search & Filter */}
      <div className="flex gap-4">
        <input
          type="text"
          placeholder="Search logs..."
          className="flex-1 border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <select className="border rounded-lg px-4 py-2">
          <option>All Actions</option>
          <option>Violations</option>
          <option>Payments</option>
          <option>Authentication</option>
        </select>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-6">
        <HistoryStatCard
          label="Total Logs"
          value={historySummary.totalLogs}
          icon="🕒"
        />
        <HistoryStatCard
          label="Today's Activity"
          value={historySummary.todayActivity}
          icon="⏰"
        />
        <HistoryStatCard
          label="Active Users"
          value={historySummary.activeUsers}
          icon="👤"
        />
      </div>

      {/* Timeline */}
      <div className="bg-white border rounded-xl p-6">
        <h2 className="text-xl font-semibold mb-4">
          Activity Timeline
        </h2>

        <div className="space-y-4">
          {activityLogs.map((log) => (
            <TimelineItem key={log.id} log={log} />
          ))}
        </div>
      </div>
    </div>
  );
}
