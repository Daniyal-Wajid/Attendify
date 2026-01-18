import Topbar from "../../components/Topbar";
import PolicyRulesGrid from "../../components/PolicyRulesGrid";
import { Play } from "lucide-react";

export default function PolicyRules() {
  return (
    <>
      <Topbar />

      <div className="p-6 space-y-6 bg-gray-50 dark:bg-gray-900 min-h-full">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Policy Rules</h1>
            <p className="text-slate-500 dark:text-gray-400">
              Manage violation policies and penalties
            </p>
          </div>

          <div className="flex gap-3">
            <button className="flex items-center gap-2 border px-4 py-2 rounded-lg hover:bg-slate-50">
              <Play size={16} />
              Simulate Rule
            </button>

            <button className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700">
              + Add Rule
            </button>
          </div>
        </div>

        <PolicyRulesGrid />
      </div>
    </>
  );
}
