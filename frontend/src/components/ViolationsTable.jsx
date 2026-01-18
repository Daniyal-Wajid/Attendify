import Badge from "./Badge";
import { Eye } from "lucide-react";
import { violations } from "../data/mockViolations";

export default function ViolationsTable() {
  return (
    <div className="bg-white rounded-xl border">
      <div className="flex justify-between items-center p-6">
        <h2 className="text-lg font-semibold">Recent Violations</h2>
        <button className="text-sm px-4 py-2 border rounded-lg hover:bg-slate-100">
          View All
        </button>
      </div>

      <table className="w-full text-sm">
        <thead className="text-slate-500 border-t border-b">
          <tr>
            <th className="text-left p-4">ID</th>
            <th className="text-left p-4">Student</th>
            <th className="text-left p-4">Type</th>
            <th className="text-left p-4">Location</th>
            <th className="text-left p-4">Confidence</th>
            <th className="text-left p-4">Status</th>
            <th className="text-left p-4">Time</th>
            <th className="text-center p-4">Action</th>
          </tr>
        </thead>

        <tbody>
          {violations.map((v) => (
            <tr key={v.id} className="border-b hover:bg-slate-50">
              <td className="p-4 font-medium">{v.id}</td>
              <td className="p-4 text-slate-600 italic">{v.student}</td>
              <td className="p-4">
                {v.type}{" "}
                <Badge text={v.severity} variant={v.severity} />
              </td>
              <td className="p-4">{v.location}</td>
              <td className="p-4">{v.confidence}</td>
              <td className="p-4">
                <Badge text={v.status} variant={v.status} />
              </td>
              <td className="p-4 text-slate-500">{v.time}</td>
              <td className="p-4 text-center">
                <Eye className="inline cursor-pointer text-slate-500 hover:text-blue-600" />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
