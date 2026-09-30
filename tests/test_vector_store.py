"""v9.78: 向量库加载鲁棒性测试 —— 候选回退序 + 全部失败可行动报错"""

import pytest
from unittest.mock import patch

from app.rag.vector_store import _candidate_kb_dirs


class TestCandidateKbDirs:
    """候选目录有序生成（活跃指针 → 最新 medical_kb_v* → 顶层 legacy）"""

    def _patch_base(self, tmp_path, monkeypatch):
        # 把 config.PERSIST_DIRECTORY 指向临时目录，避免读到真实 data/chroma_db
        import app.rag.vector_store as vsmod
        monkeypatch.setattr(vsmod.config, "PERSIST_DIRECTORY", tmp_path)

    def test_active_first(self, tmp_path, monkeypatch):
        self._patch_base(tmp_path, monkeypatch)
        active_dir = str(tmp_path / "medical_kb_vb1")
        (tmp_path / "medical_kb_vb1").mkdir(parents=True)
        (tmp_path / "medical_kb_va1").mkdir()
        # 多个世代：最新 mtime 应排在前，但活跃指针永远第一
        cands = _candidate_kb_dirs(active_dir=active_dir, active_name="medical_kb_vb1")
        assert cands[0] == (active_dir, "medical_kb_vb1")
        # 活跃目录不重复出现在候选里
        dirs = [c for c, _ in cands]
        assert dirs.count(active_dir) == 1

    def test_siblings_by_mtime_with_name(self, tmp_path, monkeypatch):
        self._patch_base(tmp_path, monkeypatch)
        # 无活跃指针 → 世代目录按 mtime 倒序，且带自身名作 collection_name
        (tmp_path / "medical_kb_vold").mkdir()
        (tmp_path / "medical_kb_vnew").mkdir()
        new = tmp_path / "medical_kb_vnew"
        import os, time
        os.utime(new, (time.time(), time.time() + 10))  # 让 new 的 mtime 更新
        cands = _candidate_kb_dirs(None, None)
        # 只取以 medical_kb_ 开头的候选，断言 new 先于 old
        sub = [c for c, n in cands if "medical_kb_v" in str(c)]
        assert sub[0] == str(new)

    def test_legacy_always_last(self, tmp_path, monkeypatch):
        self._patch_base(tmp_path, monkeypatch)
        (tmp_path / "medical_kb_vx").mkdir()
        cands = _candidate_kb_dirs(None, None)
        # 最后一个必是顶层 legacy（PERSIST_DIRECTORY），collection_name=None
        assert cands[-1][1] is None


class TestCreateVectorStoreFallback:
    """create_vector_store 加载分支：活跃目录坏 → 回退到下一候选；全失败抛可行动错误"""

    def _make_manager(self, legacy_dir):
        from app.rag.vector_store import VectorStoreManager
        return VectorStoreManager(persist_directory=str(legacy_dir))

    def test_falls_back_to_viable_sibling(self, tmp_path, monkeypatch):
        import app.rag.vector_store as vsmod
        # PERSIST_DIRECTORY 本身即 chroma_db 底座：兄弟世代是其直接子目录
        monkeypatch.setattr(vsmod.config, "PERSIST_DIRECTORY", tmp_path)
        legacy = tmp_path
        legacy.mkdir(exist_ok=True)
        # 兄弟候选目录：医疗 kb 世代，位于 PERSIST_DIRECTORY 直接子目录、可探测成功
        sibling = tmp_path / "medical_kb_vgood"
        sibling.mkdir(parents=True)

        manager = self._make_manager(legacy)
        fake_vs = object()

        # 活跃指针目录损坏 → 探测返回 None；兄弟目录 good → 探测成功返回 fake_vs
        monkeypatch.setattr(
            "app.rag.vector_store._resolve_active_collection",
            lambda: (str(tmp_path / "zz_broken_active"), "medical_kb_vbad"),
        )

        def _probe(path, name, emb):
            if "broken_active" in str(path):
                return None
            return fake_vs

        monkeypatch.setattr("app.rag.vector_store._probe_chroma", _probe)
        vs = manager.create_vector_store([])
        assert vs is fake_vs
        # 生效目录被同步为回退到的兄弟目录
        assert str(manager.persist_directory).endswith("medical_kb_vgood")

    def test_all_fail_raises_actionable(self, tmp_path, monkeypatch):
        legacy = tmp_path / "chroma_db"
        legacy.mkdir()
        manager = self._make_manager(legacy)

        monkeypatch.setattr(
            "app.rag.vector_store._resolve_active_collection", lambda: (None, None)
        )
        monkeypatch.setattr(
            "app.rag.vector_store._probe_chroma", lambda p, n, e: None
        )
        with pytest.raises(RuntimeError) as ei:
            manager.create_vector_store([])
        assert "重建向量库" in str(ei.value)