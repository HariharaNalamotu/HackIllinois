import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { HomePage } from './pages/HomePage';
import { EditorPage } from './pages/EditorPage';
import { InferencePage } from './pages/InferencePage';
import { TestPage } from './pages/TestPage';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/editor/:id" element={<EditorPage />} />
        <Route path="/run/:id" element={<InferencePage />} />
        <Route path="/test/:id" element={<TestPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
