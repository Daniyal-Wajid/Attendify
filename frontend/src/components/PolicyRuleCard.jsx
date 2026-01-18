import { Pencil } from "lucide-react";
import SeverityBadge from "./SeverityBadge";

export default function PolicyRuleCard({ rule }) {
  return (
    <div className="bg-white dark:bg-gray-800 border dark:border-gray-700 rounded-xl p-6 flex flex-col justify-between shadow-sm">
      <div className="space-y-3">
        <div className="flex justify-between items-center">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">{rule.title}</h3>
          <SeverityBadge level={rule.severity} />
        </div>

        <div className="text-sm text-slate-500 dark:text-gray-400 flex justify-between">
          <span>Rule ID</span>
          <span className="font-medium text-slate-700 dark:text-gray-300">{rule.id}</span>
        </div>

        <div className="text-sm text-slate-500 dark:text-gray-400 flex justify-between">
          <span>Penalty Amount</span>
          <span className="text-blue-600 dark:text-blue-400 font-bold">
            Rs. {rule.penalty}
          </span>
        </div>
      </div>

      <button className="mt-6 flex items-center justify-center gap-2 border dark:border-gray-600 rounded-lg py-2 hover:bg-slate-50 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 transition-colors">
        <Pencil size={16} />
        Edit Rule
      </button>
    </div>
  );
}
