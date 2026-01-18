// src/pages/Dashboard.jsx
import { useEffect, useState } from 'react'

export default function Dashboard() {
  const [data, setData] = useState(null)

  useEffect(() => {
    fetch('http://localhost:5000/api/dashboard')
      .then(res => res.json())
      .then(setData)
  }, [])

  if (!data) return <div className="p-6">Loading...</div>

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <h1 className="text-3xl font-bold mb-6">Attendance Dashboard</h1>
      <div className="grid grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-xl shadow">
          <p className="text-gray-500">Total Students</p>
          <h2 className="text-4xl font-bold">{data.students}</h2>
        </div>
      </div>

      <div className="mt-8 bg-white rounded-xl shadow p-6">
        <h2 className="text-xl font-semibold mb-4">Attendance Records</h2>
        <table className="w-full text-left">
          <thead>
            <tr className="border-b">
              <th>Name</th>
              <th>Date</th>
            </tr>
          </thead>
          <tbody>
            {data.attendance.map(a => (
              <tr key={a._id} className="border-b">
                <td>{a.student?.name}</td>
                <td>{new Date(a.date).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
