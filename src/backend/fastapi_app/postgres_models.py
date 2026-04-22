from __future__ import annotations

from pgvector.sqlalchemy import Vector
from sqlalchemy import Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Capability(Base):
    __tablename__ = "capabilities"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    subcategory_name: Mapped[str] = mapped_column()
    subcategory_description: Mapped[str] = mapped_column()
    classification_name: Mapped[str] = mapped_column()
    category_name: Mapped[str] = mapped_column()
    naics_code: Mapped[str] = mapped_column(nullable=True)
    embedding_3l: Mapped[Vector] = mapped_column(Vector(1024), nullable=True)
    embedding_nomic: Mapped[Vector] = mapped_column(Vector(768), nullable=True)

    def to_dict(self, include_embedding: bool = False):
        model_dict = {column.name: getattr(self, column.name) for column in self.__table__.columns}
        if not include_embedding:
            del model_dict["embedding_3l"]
            del model_dict["embedding_nomic"]
        return model_dict

    def to_str_for_rag(self):
        return (
            f"Classification:{self.classification_name} "
            f"Category:{self.category_name} "
            f"Subcategory:{self.subcategory_name} "
            f"Description:{self.subcategory_description} "
            f"NAICS:{self.naics_code}"
        )

    def to_str_for_embedding(self):
        return (
            f"Classification: {self.classification_name} "
            f"Category: {self.category_name} "
            f"Subcategory: {self.subcategory_name} "
            f"Description: {self.subcategory_description}"
        )


table_name = Capability.__tablename__

index_3l = Index(
    f"hnsw_index_for_cosine_{table_name}_embedding_3l",
    Capability.embedding_3l,
    postgresql_using="hnsw",
    postgresql_with={"m": 16, "ef_construction": 64},
    postgresql_ops={"embedding_3l": "vector_cosine_ops"},
)

index_nomic = Index(
    f"hnsw_index_for_cosine_{table_name}_embedding_nomic",
    Capability.embedding_nomic,
    postgresql_using="hnsw",
    postgresql_with={"m": 16, "ef_construction": 64},
    postgresql_ops={"embedding_nomic": "vector_cosine_ops"},
)
