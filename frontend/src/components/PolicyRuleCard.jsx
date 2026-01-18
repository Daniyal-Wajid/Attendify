import { Pencil } from "lucide-react";
import SeverityBadge from "./SeverityBadge";

export default function PolicyRuleCard({ rule }) {
  return (
    <div className="bg-white border rounded-xl p-6 flex flex-col justify-between">
      <div className="space-y-3">
        <div className="flex justify-between items-center">
          <h3 className="text-lg font-semibold">{rule.title}</h3>
          <SeverityBadge level={rule.severity} />
        </div>

        <div className="text-sm text-slate-500 flex justify-between">
          <span>Rule ID</span>
          <span className="font-medium text-slate-700">{rule.id}</span>
        </div>

        <div className="text-sm text-slate-500 flex justify-between">
          <span>Penalty Amount</span>
          <span className="text-blue-600 font-bold">
            Rs. {rule.penalty}
          </span>
        </div>
      </div>

      <button className="mt-6 flex items-center justify-center gap-2 border rounded-lg py-2 hover:bg-slate-50">
        <Pencil size={16} />
        Edit Rule
      </button>
    </div>
  );
}
