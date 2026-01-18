import Topbar from "../../components/Topbar";
import ViolationsFilters from "../../components/ViolationsFilters";
import ViolationsTableFull from "../../components/ViolationsTableFull";

export default function Violations() {
  return (
    <>
      <Topbar />

      <div className="p-6 space-y-6">
        <div>
          <h1 className="text-2xl font-bold">Violations</h1>
          <p className="text-slate-500">
            Manage and review all violations
          </p>
        </div>

        <ViolationsFilters />
        <ViolationsTableFull />
      </div>
    </>
  );
}
