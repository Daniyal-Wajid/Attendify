import Topbar from "../../components/Topbar";
import StudentsFilters from "../../components/StudentsFilters";
import StudentsGrid from "../../components/StudentsGrid";

export default function Students() {
  return (
    <>
      <Topbar />

      <div className="p-6 space-y-6 bg-gray-50 dark:bg-gray-900 min-h-full">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Students</h1>
          <p className="text-slate-500 dark:text-gray-400">
            View and manage student records
          </p>
        </div>

        <StudentsFilters />
        <StudentsGrid />
      </div>
    </>
  );
}
