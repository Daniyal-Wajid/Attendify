import Topbar from "../../components/Topbar";
import StudentsFilters from "../../components/StudentsFilters";
import StudentsGrid from "../../components/StudentsGrid";

export default function Students() {
  return (
    <>
      <Topbar />

      <div className="p-6 space-y-6">
        <div>
          <h1 className="text-2xl font-bold">Students</h1>
          <p className="text-slate-500">
            View and manage student records
          </p>
        </div>

        <StudentsFilters />
        <StudentsGrid />
      </div>
    </>
  );
}
