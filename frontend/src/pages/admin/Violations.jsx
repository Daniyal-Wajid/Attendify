import Topbar from "../../components/Topbar";
import ViolationsFilters from "../../components/ViolationsFilters";
import ViolationsTableFull from "../../components/ViolationsTableFull";

export default function Violations() {
  return (
    <>
      <Topbar />

      <div className="p-6 space-y-6 bg-gray-50 dark:bg-gray-900 min-h-full">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Violations</h1>
          <p className="text-slate-500 dark:text-gray-400">
            Manage and review all violations
          </p>
        </div>

        <ViolationsFilters />
        <ViolationsTableFull />
      </div>
    </>
  );
}
