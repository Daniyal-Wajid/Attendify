import {
  analyticsSummary,
  violationsTrend,
  severityDistribution,
} from "../../data/mockAnalytics";

import AnalyticsStatCard from "../../components/AnalyticsStatCard";
import ViolationsTrendChart from "../../components/ViolationsTrendChart";
import SeverityDonutChart from "../../components/SeverityDonutChart";
import { Download } from "lucide-react";

export default function Analytics() {
  return (
    <div className="p-6 space-y-6 bg-gray-50 dark:bg-gray-900 min-h-full">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Analytics & Reports</h1>
          <p className="text-slate-500 dark:text-gray-400">
            View insights and generate reports
          </p>
        </div>

        <button className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700">
          <Download size={16} />
          Download Report
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-6">
        <AnalyticsStatCard
          label="Total Violations"
          value={analyticsSummary.totalViolations}
          icon="⚠️"
        />
        <AnalyticsStatCard
          label="Total Penalties"
          value={`Rs. ${analyticsSummary.totalPenalties}`}
          icon="💰"
        />
        <AnalyticsStatCard
          label="Collected"
          value={`Rs. ${analyticsSummary.collected}`}
          icon="📈"
        />
        <AnalyticsStatCard
          label="Pending"
          value={`Rs. ${analyticsSummary.pending}`}
          icon="📉"
        />
      </div>

      {/* Tabs */}
      <div className="flex gap-2">
        {["Daily", "Weekly", "Monthly"].map((t) => (
          <button
            key={t}
            className={`px-4 py-2 rounded-lg border ${
              t === "Daily"
                ? "bg-blue-50 border-blue-500 text-blue-600"
                : "hover:bg-slate-50"
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-2 gap-6">
        <ViolationsTrendChart data={violationsTrend} />
        <SeverityDonutChart data={severityDistribution} />
      </div>
    </div>
  );
}
