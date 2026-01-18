import { Eye } from "lucide-react";
import Badge from "./Badge";
import { violations } from "../data/mockViolationsFull";

export default function ViolationsTableFull() {
  return (
    <div className="bg-white border rounded-xl overflow-hidden">
      <table className="w-full text-sm">
        <thead className="bg-slate-50 text-slate-600 border-b">
          <tr>
            <th className="p-4 text-left">ID</th>
            <th className="p-4 text-left">Student</th>
            <th className="p-4 text-left">Type</th>
            <th className="p-4 text-left">Severity</th>
            <th className="p-4 text-left">Confidence</th>
            <th className="p-4 text-left">Location</th>
            <th className="p-4 text-left">Camera</th>
            <th className="p-4 text-left">Time</th>
            <th className="p-4 text-left">Status</th>
            <th className="p-4 text-center">Actions</th>
          </tr>
        </thead>

        <tbody>
          {violations.map((v) => (
            <tr key={v.id} className="border-b hover:bg-slate-50">
              <td className="p-4 font-medium">{v.id}</td>

              <td className="p-4">
                <div className="font-medium">{v.student}</div>
                {v.studentId && (
                  <div className="text-xs text-slate-500">{v.studentId}</div>
                )}
              </td>

              <td className="p-4">{v.type}</td>

              <td className="p-4">
                <Badge text={v.severity} variant={v.severity} />
              </td>

              <td className="p-4">{v.confidence}</td>
              <td className="p-4">{v.location}</td>
              <td className="p-4">{v.camera}</td>
              <td className="p-4 text-slate-500">{v.time}</td>

              <td className="p-4">
                <Badge text={v.status} variant={v.status} />
              </td>

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
