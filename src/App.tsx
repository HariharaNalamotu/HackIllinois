import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { HomePage } from './pages/HomePage';
import { EditorPage } from './pages/EditorPage';
import { TestPage } from './pages/TestPage';
import { RLHFPage } from './pages/RLHFPage';
import { RLAIFPage } from './pages/RLAIFPage';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/editor/:id" element={<EditorPage />} />
        <Route path="/test/:id" element={<TestPage />} />
        <Route path="/rlhf/:id" element={<RLHFPage />} />
        <Route path="/rlaif/:id" element={<RLAIFPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
